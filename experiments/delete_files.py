import os
import shutil
from pathlib import Path


def delete_files_by_name(root_dir, target_names, include_subdirs=True, dry_run=True):
    """
    递归删除指定目录下所有名为 target_names 的文件或文件夹。

    :param root_dir: 要搜索的根目录路径
    :param target_names: 要删除的文件/文件夹名列表（字符串列表，如 ['__pycache__', '.DS_Store']）
    :param include_subdirs: 是否递归进入子目录
    :param dry_run: 如果为 True，只打印将要删除的项，不实际删除
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"错误：目录不存在 -> {root_path}")
        return

    if isinstance(target_names, str):
        target_names = [target_names]  # 允许传入单个字符串

    count = 0
    for item in root_path.rglob('*') if include_subdirs else root_path.iterdir():
        if item.name in target_names:
            try:
                if item.is_file():
                    if dry_run:
                        print(f"[文件] 将删除: {item}")
                    else:
                        item.unlink()
                        print(f"[文件] 已删除: {item}")
                elif item.is_dir():
                    if dry_run:
                        print(f"[目录] 将删除: {item}")
                    else:
                        if item.is_symlink():
                            item.unlink()
                        else:
                            shutil.rmtree(item)
                        print(f"[目录] 已删除: {item}")
                count += 1
            except Exception as e:
                print(f"删除失败: {item} - 错误: {e}")

    action = "（模拟）" if dry_run else ""
    print(f"\n总共处理了 {count} 个名为 {target_names} 的项目{action}。")


# =============================
#        使用示例
# =============================

if __name__ == "__main__":
    # 设置参数
    directory = "."  # 要清理的目录，"." 表示当前目录
    filenames_to_delete = [
        'config.json',
        'global_model.pth'
    ]

    # 🔥 修改 dry_run=False 来真正执行删除！
    delete_files_by_name(
        root_dir=directory,
        target_names=filenames_to_delete,
        include_subdirs=True,
        dry_run=True  # 👈 设为 False 才会真实删除
    )