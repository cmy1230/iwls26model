"""
批量贝叶斯优化搜索：对目录下所有 BLIF 文件并行运行 bayesian_search.py，
结果汇总到一个 CSV 文件。

用法:
    python batch_search.py --abc_exe ./abc --blif_dir ./blif_files [选项]

示例:
    # 基础 AIG 优化，4 路并行
    python batch_search.py --abc_exe ./abc --blif_dir ./data/blif --parallel 4

    # SCL 映射优化
    python batch_search.py --abc_exe ./abc --blif_dir ./data/blif --parallel 4 \
        --mapping SCL --cell_lib "lib1.lib;lib2.lib"

    # 透传更多参数给 bayesian_search.py
    python batch_search.py --abc_exe ./abc --blif_dir ./data/blif --parallel 8 \
        --extra_args "--seq_len 25 --n_trials 500 --optimize area --n_jobs 2"
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def run_one(blif_path: str, abc_exe: str, output_dir: str,
            mapping: str, map_arg: str, cell_lib: str,
            extra_args: list, script_path: str) -> dict:
    """对单个 BLIF 文件运行 bayesian_search.py，返回结果字典"""
    stem = Path(blif_path).stem
    json_out = str(Path(output_dir) / f"bayesian_result_{stem}.json")

    cmd = [
        sys.executable, script_path,
        "--abc_exe", abc_exe,
        "--input_file", blif_path,
        "--output", json_out,
    ]
    if mapping:
        cmd += ["--mapping", mapping]
    if map_arg:
        cmd += ["--map_arg", map_arg]
    if cell_lib:
        cmd += ["--cell_lib", cell_lib]
    cmd += extra_args

    t0 = time.time()
    print(f"[START] {stem}")

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=36000,
        )
        elapsed = time.time() - t0

        if proc.returncode != 0:
            print(f"[FAIL]  {stem}  ({elapsed:.0f}s)  returncode={proc.returncode}")
            last_lines = "\n".join(proc.stdout.strip().split("\n")[-5:])
            print(f"        {last_lines}")
            return {"circuit": stem, "status": "FAIL", "error": last_lines}

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"[TIMEOUT] {stem}  ({elapsed:.0f}s)")
        return {"circuit": stem, "status": "TIMEOUT"}
    except Exception as e:
        print(f"[ERROR] {stem}: {e}")
        return {"circuit": stem, "status": "ERROR", "error": str(e)}

    if not Path(json_out).exists():
        print(f"[FAIL]  {stem}  JSON 结果文件未生成")
        return {"circuit": stem, "status": "FAIL", "error": "no json output"}

    with open(json_out, "r", encoding="utf-8") as f:
        data = json.load(f)

    best = data.get("best_stats", {})
    init = data.get("init_stats", {})
    row = {
        "circuit": stem,
        "status": "OK",
        "init_nodes": init.get("nodes", ""),
        "init_levels": init.get("levels", ""),
        "init_area": init.get("area", ""),
        "init_delay": init.get("delay", ""),
        "best_nodes": best.get("nodes", ""),
        "best_levels": best.get("levels", ""),
        "best_area": best.get("area", ""),
        "best_delay": best.get("delay", ""),
        "best_cost": data.get("best_cost", ""),
        "nodes_improve": data.get("improvement", {}).get("nodes", ""),
        "levels_improve": data.get("improvement", {}).get("levels", ""),
        "area_improve": data.get("improvement", {}).get("area", ""),
        "delay_improve": data.get("improvement", {}).get("delay", ""),
        "best_sequence": data.get("best_sequence_str", ""),
        "abc_verify_cmd": data.get("abc_verify_cmd", ""),
        "n_trials": data.get("n_trials", ""),
        "elapsed_sec": f"{elapsed:.1f}",
    }

    print(f"[DONE]  {stem}  ({elapsed:.0f}s)  "
          f"nodes={best.get('nodes','')}  area={best.get('area','')}  "
          f"delay={best.get('delay','')}  cost={data.get('best_cost',''):.6f}")
    return row


def main():
    parser = argparse.ArgumentParser(
        description="批量对目录下所有 BLIF 文件运行贝叶斯优化搜索",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--abc_exe", type=str, required=True,
                        help="ABC 可执行文件路径")
    parser.add_argument("--blif_dir", type=str, required=True,
                        help="BLIF 文件所在目录")
    parser.add_argument("--output_csv", type=str, default="",
                        help="汇总结果 CSV 路径 (default: batch_result_<目录名>.csv)")
    parser.add_argument("--output_dir", type=str, default="",
                        help="各电路 JSON 结果存放目录 (default: 与 CSV 同目录下 json_results/)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="同时处理的 BLIF 文件数 (default: 1)")
    parser.add_argument("--mapping", type=str, default="",
                        choices=["", "FPGA", "SCL"],
                        help="技术映射方式，透传给 bayesian_search.py")
    parser.add_argument("--map_arg", type=str, default="",
                        help="映射参数，透传给 bayesian_search.py")
    parser.add_argument("--cell_lib", type=str, default="",
                        help="标准单元库路径，透传给 bayesian_search.py")
    parser.add_argument("--extra_args", type=str, default="",
                        help="额外参数，原样透传给 bayesian_search.py (如 \"--seq_len 25 --n_trials 500\")")
    parser.add_argument("--script", type=str, default="",
                        help="bayesian_search.py 脚本路径 (default: 与本脚本同目录)")

    args = parser.parse_args()

    blif_dir = Path(args.blif_dir).resolve()
    if not blif_dir.is_dir():
        print(f"错误: 目录不存在: {blif_dir}")
        sys.exit(1)

    blif_files = sorted(blif_dir.glob("*.blif"))
    if not blif_files:
        print(f"错误: {blif_dir} 下没有 .blif 文件")
        sys.exit(1)

    script_path = args.script
    if not script_path:
        script_path = str(Path(__file__).resolve().parent / "bayesian_search.py")
    if not Path(script_path).exists():
        print(f"错误: bayesian_search.py 不存在: {script_path}")
        sys.exit(1)

    output_csv = args.output_csv
    if not output_csv:
        output_csv = f"batch_result_{blif_dir.name}_new.csv"

    output_dir = args.output_dir
    if not output_dir:
        output_dir = str(Path(output_csv).resolve().parent / "json_results")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    extra_args = args.extra_args.split() if args.extra_args else []
    parallel = max(1, args.parallel)

    print(f"{'='*60}")
    print(f"批量贝叶斯优化搜索")
    print(f"{'='*60}")
    print(f"BLIF 目录:   {blif_dir}")
    print(f"文件数量:    {len(blif_files)}")
    print(f"并行数:      {parallel}")
    print(f"映射方式:    {args.mapping or '无 (仅 AIG 优化)'}")
    if args.cell_lib:
        print(f"单元库:      {args.cell_lib}")
    if extra_args:
        print(f"额外参数:    {' '.join(extra_args)}")
    print(f"结果 CSV:    {output_csv}")
    print(f"JSON 目录:   {output_dir}")
    print(f"{'='*60}\n")

    csv_fields = [
        "circuit", "status",
        "init_nodes", "init_levels", "init_area", "init_delay",
        "best_nodes", "best_levels", "best_area", "best_delay",
        "best_cost",
        "nodes_improve", "levels_improve", "area_improve", "delay_improve",
        "best_sequence", "abc_verify_cmd", "n_trials", "elapsed_sec",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()

    t_start = time.time()
    results = []

    def _flush_csv():
        sorted_rows = sorted(results, key=lambda r: r.get("circuit", ""))
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            writer.writeheader()
            for row in sorted_rows:
                writer.writerow(row)

    with ProcessPoolExecutor(max_workers=parallel) as pool:
        futures = {}
        for bf in blif_files:
            fut = pool.submit(
                run_one,
                str(bf), args.abc_exe, output_dir,
                args.mapping, args.map_arg, args.cell_lib,
                extra_args, script_path,
            )
            futures[fut] = bf.stem

        for fut in as_completed(futures):
            row = fut.result()
            results.append(row)
            _flush_csv()
            print(f"  [{len(results)}/{len(blif_files)}] CSV 已更新")

    elapsed = time.time() - t_start
    n_ok = sum(1 for r in results if r.get("status") == "OK")
    n_fail = len(results) - n_ok

    print(f"\n{'='*60}")
    print(f"批量搜索完成!")
    print(f"{'='*60}")
    print(f"总耗时:      {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"成功/失败:   {n_ok}/{n_fail}")
    print(f"结果已保存:  {output_csv}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
