import pickle
from Bio import SeqIO
from gensim.models.doc2vec import Doc2Vec
import warnings
warnings.filterwarnings("ignore")


def load_protein_encodings(encoding_file):
    """
    加载蛋白质编码文件
    """
    with open(encoding_file, 'rb') as f:
        protein_encodings = pickle.load(f)
    return protein_encodings


def extract_features(model_file, protein_encodings, fasta_file, output_file):
    """
    根据 FASTA 文件中的序列 ID 提取蛋白质的特征并保存到文件中
    """
    # 加载 Doc2Vec 模型
    model = Doc2Vec.load(model_file)

    feature_dict = {}
    for seq_record in SeqIO.parse(fasta_file, "fasta"):
        sequence_id = seq_record.id
        if sequence_id in protein_encodings:
            feature_vector = protein_encodings[sequence_id]
        else:
            # 如果序列不在特征文件中，则用模型进行推断
            inferred_vector = model.infer_vector(seq_record.seq)
            feature_vector = list(inferred_vector)
        feature_dict[sequence_id] = feature_vector

    # 保存特征字典到文件中
    with open(output_file, 'wb') as f_out:
        pickle.dump(feature_dict, f_out)


if __name__ == "__main__":
    # 模型文件
    model_file = 'doc2vec/human_virus_all-doc2vector-all-5-2-32-3-70_0-5000_HVPPI_model.pkl'

    # 加载蛋白质编码文件
    encoding_file = 'doc2vec/human_virus_all-doc2vector-all-5-2-32-3-70_0-5000_HVPPI.pkl'
    protein_encodings = load_protein_encodings(encoding_file)

    # 蛋白质序列的 FASTA 文件
    fasta_file = '../../data/v_and_h.fasta'

    # 输出文件路径
    output_file = 'v_and_h_doc2vec.pkl'

    # 提取特征并保存到文件中
    extract_features(model_file, protein_encodings, fasta_file, output_file)
