import os, sys

def add_prefix_subfolders():
    mark = '20-'  # Prefix content ready to be added
    old_names = os.listdir(path)
    old_names.sort(key=lambda x: int(x.split('.')[0]))
    i = 1
    for old_name in old_names:
        if old_name != sys.argv[0]:
            name = str(i)
            os.rename(os.path.join(path, old_name), os.path.join(path, mark + name + '.png'))
            print(old_name, "has been renamed successfully! New name is: ", mark + name + '.png')
            i = i + 1


if __name__ == '__main__':
    path = r'E:/CWLD_model/data/label/'
    add_prefix_subfolders()
