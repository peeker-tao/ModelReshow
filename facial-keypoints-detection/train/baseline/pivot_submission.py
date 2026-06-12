"""
将 submission CSV（RowId, Location）与 IdLookupTable（RowId, ImageId, FeatureName）
合并并透视为宽表格式：每行一个 ImageId，30 列（15 个关键点 × x/y），
缺失值用 "none" 填充。

输出格式：
ImageId, left_eye_center_x, left_eye_center_y, ..., right_eyebrow_outer_end_y
"""

import pandas as pd
import argparse
from pathlib import Path

# 脚本自身所在目录，所有默认相对路径都基于此解析
_SCRIPT_DIR = Path(__file__).resolve().parent


def pivot_submission(
    submission_path: str,
    lookup_path: str,
    output_path: str,
):
    sub = pd.read_csv(submission_path)
    lookup = pd.read_csv(lookup_path)

    # 按 RowId 合并预测值（两边都有 Location 列，会产生 Location_x/_y 后缀）
    merged = lookup.merge(sub, on="RowId", how="left")
    merged["Location"] = merged["Location_y"].fillna("-1")

    # 透视：ImageId × FeatureName → 30 列
    pivot = merged.pivot_table(
        index="ImageId",
        columns="FeatureName",
        values="Location",
        aggfunc="first",
    )
    pivot = pivot.reset_index()
    pivot.columns.name = None

    # 固定列顺序，缺失的 FeatureName（标注表本身缺少的行）用 "none" 填充
    feature_names = sorted(merged["FeatureName"].unique())
    cols = ["ImageId"] + feature_names
    pivot = pivot[cols]
    pivot[feature_names] = pivot[feature_names].fillna("-1")

    pivot.to_csv(output_path, index=False)
    print(f"✓ 已保存: {output_path}")
    print(f"  图像数: {len(pivot)}, 特征列数: {len(feature_names)}")


def main():
    parser = argparse.ArgumentParser(
        description="将 submission（RowId→Location）转换为 ImageId×30 的宽表 CSV",
    )
    parser.add_argument(
        "--submission",
        default="output/submission8_1.85.csv",
        help="submission CSV 路径（默认: output/submission8_1.85.csv）",
    )
    parser.add_argument(
        "--lookup",
        default="../../data/IdLookupTable.csv",
        help="IdLookupTable CSV 路径（默认: ../../data/IdLookupTable.csv）",
    )
    parser.add_argument(
        "--output",
        default="output/submission_pivot.csv",
        help="输出 CSV 路径（默认: output/submission_pivot.csv）",
    )
    args = parser.parse_args()

    # 将相对路径基于脚本所在目录解析，而非当前工作目录
    def _resolve(p: str) -> str:
        path = Path(p)
        if path.is_absolute():
            return p
        return str(_SCRIPT_DIR / path)

    pivot_submission(
        submission_path=_resolve(args.submission),
        lookup_path=_resolve(args.lookup),
        output_path=_resolve(args.output),
    )


if __name__ == "__main__":
    main()
