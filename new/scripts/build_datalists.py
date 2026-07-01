"""生成 datalist/{train,validate,test}.txt 数据划分清单。

只有更换数据集或想重新划分训练/验证集时才需要运行。
默认 seed=123, val_ratio=0.05。运行目录为 new/。

用法:
  python scripts/build_datalists.py [--val-ratio 0.05] [--seed 123]
"""
import argparse
import random
from pathlib import Path


def relative_model_dir(path: Path, root: Path, suffix_depth: int) -> str:
    model_dir = path
    for _ in range(suffix_depth):
        model_dir = model_dir.parent
    return model_dir.relative_to(root).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", default="../milestone/dataset_train")
    parser.add_argument("--test-root", default="../milestone/dataset_test_noisy")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", default="../milestone/datalist")
    args = parser.parse_args()

    train_root = Path(args.train_root)
    test_root = Path(args.test_root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    train_paths = sorted(train_root.glob("shapenet/*/*/models/model_normalized.obj"))
    test_paths = sorted(test_root.glob("shapenet/*/*/noisy.npy"))
    if not train_paths:
        raise FileNotFoundError(f"no training OBJ files found under {train_root}")
    if not test_paths:
        raise FileNotFoundError(f"no noisy.npy files found under {test_root}")

    # 训练集: 去掉 models/model_normalized.obj 两级得到 model_id
    # 测试集: 去掉 noisy.npy 一级得到 model_id
    train_items = [relative_model_dir(p, train_root, 2) for p in train_paths]
    test_items = [relative_model_dir(p, test_root, 1) for p in test_paths]

    random.Random(args.seed).shuffle(train_items)
    val_count = max(1, int(len(train_items) * args.val_ratio))
    validate_items = train_items[:val_count]
    training_items = train_items[val_count:]

    (output / "train.txt").write_text("\n".join(training_items) + "\n", encoding="utf-8")
    (output / "validate.txt").write_text("\n".join(validate_items) + "\n", encoding="utf-8")
    (output / "test.txt").write_text("\n".join(test_items) + "\n", encoding="utf-8")

    print(f"train={len(training_items)}, validate={len(validate_items)}, test={len(test_items)}")


if __name__ == "__main__":
    main()
