import os
import argparse
import importlib
import mmcv
from mmcv import Config
from mmdet3d.datasets import build_dataset


def import_plugins(cfg):
    if getattr(cfg, "plugin", False):
        plugin_dir = getattr(cfg, "plugin_dir", None)
        if plugin_dir is not None:
            module_path = plugin_dir.replace("/", ".")
            if module_path.endswith("."):
                module_path = module_path[:-1]
            print(f"[INFO] Importing plugin: {module_path}")
            importlib.import_module(module_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--json-prefix", required=True)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    import_plugins(cfg)

    cfg.data.test.test_mode = True

    print("[INFO] Building test dataset...")
    dataset = build_dataset(cfg.data.test)

    print(f"[INFO] Loading predictions from: {args.pkl}")
    outputs = mmcv.load(args.pkl)

    print("[INFO] Number of outputs:", len(outputs))
    print("[INFO] Number of dataset samples:", len(dataset))

    os.makedirs(args.json_prefix, exist_ok=True)

    print(f"[INFO] Converting to nuScenes JSON: {args.json_prefix}")
    result_files, tmp_dir = dataset.format_results(
        outputs,
        jsonfile_prefix=args.json_prefix
    )

    print("[DONE] result_files:")
    print(result_files)

    print("[INFO] JSON files:")
    for root, _, files in os.walk(args.json_prefix):
        for f in files:
            if f.endswith(".json"):
                print(os.path.join(root, f))


if __name__ == "__main__":
    main()
