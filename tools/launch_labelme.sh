#!/usr/bin/env bash
# =============================================================================
# launch_labelme.sh —— 启动 labelme 标注指甲分割数据（适配 labelme 7.x）
# =============================================================================
# 用法:
#   bash tools/launch_labelme.sh status                    # 查看标注进度
#   bash tools/launch_labelme.sh train onychomycosis       # 从该目录第一张开始标
#   bash tools/launch_labelme.sh train onychomycosis new   # 断点续标: 直接跳到第一张未标注的图
#   bash tools/launch_labelme.sh val  healthy
#
# 关于 labelme 7.x 的行为(与 5.x 不同, 已按实际版本核对):
#   * 自动保存默认开启, 没有 --autosave 这个参数(要关才用 --no-auto-save);
#   * 默认不把图片 base64 写进 JSON, 没有 --nodata 参数(要写才用 --with-image-data);
#   * 只接受 **一个** path 参数: 给目录就从第一张开始, 给单个文件就从那张开始并列出同目录文件;
#   * --no-sort-labels 保持 labels.txt 的类别顺序(默认是按字母排序);
#   * --validate-label exact 会把标签限制在 labels.txt 列表内, 从源头杜绝拼写错误的类别名。
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"     # nail_seg/
LABELME_BIN="$ROOT/../.venv-labelme/bin/labelme"
LABELS="$ROOT/labels.txt"
IMAGES="$ROOT/images"
SPLITS="train val"
IMG_GLOB=( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' )

die() { echo "[x] $*" >&2; exit 1; }

# labelme 7 依赖 PySide6 >= 6.8 (Qt 6.5+)，其 xcb 平台插件需要 libxcb-cursor.so.0，
# 而 Ubuntu 20.04 默认不带这个库。这里用工作区内就地解包的 .deb 提供，不需要 root、不改系统。
LIBS_DIR="$ROOT/../.libs/root/usr/lib/x86_64-linux-gnu"
if [[ -d "$LIBS_DIR" ]]; then
  export LD_LIBRARY_PATH="$LIBS_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

[[ -f "$LABELS" ]] || die "找不到 labels.txt: $LABELS (先运行 prepare_labeling.py)"

count_imgs() { find "$1" -maxdepth 1 -type f \( "${IMG_GLOB[@]}" \) | wc -l; }
count_json() { find "$1" -maxdepth 1 -type f -name '*.json' | wc -l; }

# 第一张还没有 .json 的图片; 全部标完则返回 1
first_unlabeled() {
  local dir="$1" f
  while IFS= read -r f; do
    [[ -f "$dir/${f%.*}.json" ]] && continue
    printf '%s\n' "$dir/$f"
    return 0
  done < <(find "$dir" -maxdepth 1 -type f \( "${IMG_GLOB[@]}" \) -printf '%f\n' | sort)
  return 1
}

ACTION="${1:-status}"

# ---------------------------------------------------------------- status
if [[ "$ACTION" == "status" ]]; then
  total_all=0; done_all=0
  printf '%-28s %10s %10s %8s\n' "划分/类别" "已标注" "总数" "进度"
  printf '%s\n' "--------------------------------------------------------------"
  for split in $SPLITS; do
    for dir in "$IMAGES/$split"/*/; do
      [[ -d "$dir" ]] || continue
      total=$(count_imgs "$dir"); done_n=$(count_json "$dir")
      total_all=$((total_all + total)); done_all=$((done_all + done_n))
      pct=0; [[ "$total" -gt 0 ]] && pct=$((done_n * 100 / total))
      printf '%-28s %10d %10d %7d%%\n' "$split/$(basename "$dir")" "$done_n" "$total" "$pct"
    done
  done
  printf '%s\n' "--------------------------------------------------------------"
  pct=0; [[ "$total_all" -gt 0 ]] && pct=$((done_all * 100 / total_all))
  printf '%-28s %10d %10d %7d%%\n' "合计" "$done_all" "$total_all" "$pct"
  echo
  echo "转换掩膜:  python3 $ROOT/tools/labelme2mask.py"
  exit 0
fi

# ---------------------------------------------------------------- 标注
SPLIT="$ACTION"
CLS="${2:-}"
MODE="${3:-all}"

[[ -d "$IMAGES/$SPLIT" ]] || die "没有这个划分: $SPLIT (可选: $SPLITS)"
if [[ -z "$CLS" ]]; then
  echo "请指定类别, $SPLIT 下有:"
  for dir in "$IMAGES/$SPLIT"/*/; do
    [[ -d "$dir" ]] || continue
    printf '    %-16s 已标注 %s/%s\n' "$(basename "$dir")" "$(count_json "$dir")" "$(count_imgs "$dir")"
  done
  echo
  echo "例如: bash tools/launch_labelme.sh $SPLIT $(ls "$IMAGES/$SPLIT" | head -1)"
  exit 1
fi
DIR="$IMAGES/$SPLIT/$CLS"
[[ -d "$DIR" ]] || die "没有这个类别目录: $DIR"

n_total=$(count_imgs "$DIR"); n_done=$(count_json "$DIR")

if [[ "$MODE" == "new" || "$MODE" == "--resume" ]]; then
  if ! TARGET="$(first_unlabeled "$DIR")"; then
    echo "[ok] $SPLIT/$CLS 已全部标注完成 ($n_done/$n_total)。"
    echo "     转换掩膜: python3 $ROOT/tools/labelme2mask.py"
    exit 0
  fi
  MODE_DESC="断点续标(从第一张未标注的图开始, 按 D 继续往后)"
else
  TARGET="$DIR"
  MODE_DESC="从目录第一张开始"
fi

cat <<EOF
------------------------------------------------------------------
 标注目标 : $SPLIT/$CLS        进度 $n_done/$n_total
 起始位置 : $MODE_DESC
 类别标签 : $(paste -sd' / ' "$LABELS")
 标注要求 : 用「多边形(polygon)」沿指甲边缘描一圈, 双击闭合;
            标签选「$CLS」——类别已用 --validate-label exact 锁定, 打错字会被拒绝。
            · 一张图有多个指甲 -> 每个指甲各画一个多边形
            · 背景/皮肤不要标, 掩膜里自动为 0
            · D = 下一张, A = 上一张; 已开启自动保存, 切图即存
            · 只标「整片指甲」, 不要只框病灶局部(4 类方案的类别由文件夹决定)
------------------------------------------------------------------
EOF

[[ -x "$LABELME_BIN" ]] || die "找不到 labelme: $LABELME_BIN
    请先安装:
      python3 -m venv '$ROOT/../.venv-labelme'
      '$ROOT/../.venv-labelme/bin/python' -m pip install labelme -i https://mirrors.aliyun.com/pypi/simple/"

exec "$LABELME_BIN" \
  --labels "$LABELS" \
  --no-sort-labels \
  --validate-label exact \
  "$TARGET"
