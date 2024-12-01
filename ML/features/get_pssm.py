import os
import subprocess

# 这里直接在https://ftp.ncbi.nlm.nih.gov/blast/db/下载，不需要提前建库
# 这里下载的是swissprot
# makeblastdb -in uniprot_sprot.fasta -dbtype prot -out uniprot_sprot
# 输入 fasta 文件和输出文件夹
input_file = "../../data/v_and_h.fasta"
output_folder = "pssm"

# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 初始化序列计数器
seq_count = 0


def generate_pssm(sequence, seq_id, output_file, seq_count):
    cmd = ["blast-2.15.0+/bin/psiblast.exe", "-query", "-", "-db", "swissprot_db/swissprot", "-evalue", "0.001",
           "-num_iterations", "3", "-num_threads", "6", "-out_ascii_pssm", output_file]

    try:
        # 运行 psiblast
        with subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE) as process:
            stdout, stderr = process.communicate(input=sequence.encode())
            if process.returncode != 0:
                print(f"Error occurred while running psiblast for sequence {seq_id}:")
                print(stderr.decode())
            else:
                print(f"PSSM generated for sequence {seq_count}: {seq_id}")

                # 处理生成的PSSM文件，只保留L*20的矩阵
                with open(output_file, "r") as pssm_file:
                    lines = pssm_file.readlines()
                    # 删除前3行和最后5行
                    del lines[:3]
                    del lines[-5:]
                    # 从第一列开始保留20个字符（即20个氨基酸的PSSM值）
                    processed_lines = [line.split()[2:22] for line in lines]
                # 重新写入处理后的PSSM文件
                with open(output_file, "w") as pssm_file:
                    for line in processed_lines:
                        pssm_file.write(" ".join(line) + "\n")
    except Exception as e:
        print(f"Error occurred while running psiblast for sequence {seq_id}:")
        print(str(e))


# 逐条读取 fasta 文件中的序列并生成 PSSM
with open(input_file, "r") as f:
    seq_id = None
    sequence = ""
    for fline in f:
        fline = fline.strip()
        if fline.startswith(">"):
            # 如果是 fasta 文件中的头部行，则处理之前的序列并生成 PSSM
            if seq_id is not None:
                output_file = os.path.join(output_folder, seq_id + ".pssm")
                generate_pssm(sequence, seq_id, output_file, seq_count)

            # 提取当前序列的标识符
            seq_id = fline[1:]  # 去除首尾空格
            seq_count += 1
            # 重置序列字符串
            sequence = ""
        else:
            # 如果不是头部行，则将当前行的序列添加到序列字符串中
            sequence += fline

    # 处理最后一条序列并生成 PSSM
    if seq_id is not None:
        output_file = os.path.join(output_folder, seq_id + ".pssm")
        generate_pssm(sequence, seq_id, output_file, seq_count)

# --------------------------------------
# 处理文件的空白行避免读取失败
import os


def remove_extra_lines(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 寻找第一个空白行的索引
    index = 0
    while index < len(lines) and lines[index].strip() != '':
        index += 1

    # 如果找到空白行，则保留空白行之前的内容
    if index < len(lines):
        with open(file_path, 'w') as file:
            file.writelines(lines[:index + 1])


def check_and_remove_extra_lines(folder_path):
    # 遍历文件夹中的每个文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pssm'):
            file_path = os.path.join(folder_path, file_name)
            remove_extra_lines(file_path)


# 指定文件夹路径
folder_path = "pssm"

# 检查并删除每个文件中的额外空白行
check_and_remove_extra_lines(folder_path)
