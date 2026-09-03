# -*- coding: utf-8 -*-
"""清理训练日志中的 tqdm 进度条，保留全部有效信息。

规则:
  1. 训练进度条行 (形如 "  1/300 ... 640: 12% ── 200/1476 ...") 每个
     epoch 只保留最后一行(100% 完成行)，包含该 epoch 的最终
     box/cls/dfl loss、GPU 显存与耗时。
  2. 数据集扫描进度条 (train:/val: Scanning) 只保留含 100% 的最终行。
  3. 其余行(配置、模型结构、验证指标、警告、结尾总结)原样保留。

用法:
  python tools/clean_logs.py logs/t300               # 清理目录下所有 *.out
  python tools/clean_logs.py logs/t300/t300_18580_0.out
  python tools/clean_logs.py logs/t300 --inplace     # 直接覆盖原文件(默认写 *_clean.out)
  python tools/clean_logs.py logs/t300 --summary     # 额外生成 *_summary.txt 纯摘要

清理后文件默认加 .bak 备份再覆盖(--inplace 时)。
"""
import argparse
import re
import sys
from pathlib import Path

# 训练 batch 进度条:  "      1/300      8.92G      3.66 ... 640: 45% ── 664/1476 5it/s"
# 注意: tqdm 用 \r 分隔刷新行，先按 splitlines() 切行再匹配，正则内用 [ \t] 避免跨行。
RE_TRAIN_BAR = re.compile(r'^[ \t]*\d+/\d+[ \t]+[\d.]+G?[ \t]+[\d.]+[ \t]+[\d.]+[ \t]+[\d.]+.*?:[ \t]*\d+%')
RE_EPOCH = re.compile(r'^[ \t]*(\d+/\d+)')
RE_BAR_PCT = re.compile(r'\d+%')
RE_ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')  # tqdm 行首的 \x1b[K 等转义序列


def iter_lines(path: Path):
    """按通用换行(含 \r)切行读取大文件。"""
    buf = ''
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            buf += chunk
            lines = buf.splitlines(True)
            if lines and not buf.endswith(('\n', '\r')):
                buf = lines.pop()  # 末行可能未完整，留到下轮
            else:
                buf = ''
            for line in lines:
                yield line.rstrip('\r\n') + '\n'
    if buf:
        for line in buf.splitlines():
            yield line + '\n'


def clean_file(src: Path, dst: Path) -> tuple[int, int]:
    """返回 (原行数, 保留行数)。"""
    kept = []
    last_bar = None  # 缓存当前 epoch 的最后一条进度条行
    last_ep = None   # 缓存行所属 epoch (如 '1/300')
    n_in = 0

    def flush():
        nonlocal last_bar, last_ep
        if last_bar is not None:
            kept.append(last_bar)
            last_bar, last_ep = None, None

    for raw in iter_lines(src):
        n_in += 1
        line = RE_ANSI.sub('', raw)
        if RE_TRAIN_BAR.match(line):
            ep = RE_EPOCH.match(line).group(1)
            if last_bar is not None and ep != last_ep:
                flush()  # epoch 切换，先落盘上一 epoch 的最终行
            last_bar, last_ep = line, ep
        else:
            flush()
            # 数据集 Scanning 进度条: 只留 100% 完成行
            if 'Scanning' in line and RE_BAR_PCT.search(line) and '100%' not in line:
                continue
            kept.append(line)
    flush()

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, 'w', encoding='utf-8', newline='') as f:
        f.writelines(kept)
    return n_in, len(kept)


def make_summary(src: Path, dst: Path) -> None:
    """提取纯摘要: 头部配置 + 每 epoch 完成行 + 验证结果 + 结尾。"""
    keep_kw = ('all ', 'Epoch', 'epochs completed', 'Results saved', 'mAP',
               'Image sizes', 'Starting training', 'optimizer:', 'summary:')
    lines = []
    for raw in iter_lines(src):
        line = RE_ANSI.sub('', raw)
        if RE_TRAIN_BAR.match(line) and '100%' in line:
            lines.append(line)
        elif any(k in line for k in keep_kw):
            lines.append(line)
    with open(dst, 'w', encoding='utf-8', newline='') as f:
        f.writelines(lines)


def main():
    ap = argparse.ArgumentParser(description='清理训练日志进度条')
    ap.add_argument('paths', nargs='+', help='日志文件或目录(处理目录下 *.out/*.err)')
    ap.add_argument('--inplace', action='store_true', help='直接覆盖原文件(先备份 .bak)')
    ap.add_argument('--summary', action='store_true', help='额外生成 *_summary.txt')
    args = ap.parse_args()

    files = []
    for p in args.paths:
        path = Path(p)
        if path.is_dir():
            files += [f for f in sorted(path.iterdir())
                      if f.suffix in ('.out', '.err') and f.is_file()]
        elif path.is_file():
            files.append(path)
        else:
            print(f'[跳过] 不存在: {p}', file=sys.stderr)

    if not files:
        print('没有可处理的文件', file=sys.stderr)
        return 1

    for f in files:
        old_mb = f.stat().st_size / 1e6
        if args.inplace:
            dst = f.with_suffix(f.suffix + '.tmp')
            n_in, n_out = clean_file(f, dst)
            bak = f.with_suffix(f.suffix + '.bak')
            if not bak.exists():
                f.replace(bak)
            dst.replace(f)
            new_mb = f.stat().st_size / 1e6
        else:
            dst = f.with_suffix(f.suffix + '.clean')
            n_in, n_out = clean_file(f, dst)
            new_mb = dst.stat().st_size / 1e6

        if args.summary:
            make_summary(f, f.with_suffix(f.suffix + '.summary.txt'))

        print(f'{f.name}: {n_in} -> {n_out} 行, {old_mb:.1f}MB -> {new_mb:.1f}MB')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
