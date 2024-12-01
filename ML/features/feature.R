# 特征转换

### 清除环境，设置路径
rm(list = ls())
setwd("features")

# 安装并加载
install.packages("protr")
library(protr)

# 导入FASTA文件
# 导入蛋白质序列文件
# 导入正样本,使用protcheck()函数进行氨基酸类型健全性检查，并移除非标准的序列
pos <- readFASTA("../../data/v_and_h.fasta")
length(pos)
pos <- pos[(sapply(pos, protcheck))]
length(pos)

# 提取AAC
x <- sapply(pos, extractAAC)
# 保存txt
write.table(x, file = "v_and_h_AAC.txt", sep = "\t", row.names = FALSE, quote = FALSE)

# 提取APAAC
x <- sapply(pos, extractAPAAC)
# 保存txt
write.table(x, file = "v_and_h_APAAC.txt", sep = "\t", row.names = FALSE, quote = FALSE)

# 提取CTDC
x <- sapply(pos, extractCTDC)
# 保存txt
write.table(x, file = "v_and_h_CTDC.txt", sep = "\t", row.names = FALSE, quote = FALSE)

# 提取CTDD
x <- sapply(pos, extractCTDD)
# 保存txt
write.table(x, file = "v_and_h_CTDD.txt", sep = "\t", row.names = FALSE, quote = FALSE)

# 提取CTDT
x <- sapply(pos, extractCTDT)
# 保存txt
write.table(x, file = "v_and_h_CTDT.txt", sep = "\t", row.names = FALSE, quote = FALSE)

# 提取CT
x <- sapply(pos, extractCTriad)
# 保存txt
write.table(x, file = "v_and_h_CT.txt", sep = "\t", row.names = FALSE, quote = FALSE)

# 提取DC
x <- sapply(pos, extractDC)
# 保存txt
write.table(x, file = "v_and_h_DC.txt", sep = "\t", row.names = FALSE, quote = FALSE)

# 提取Geary
x <- sapply(pos, extractGeary)
# 保存txt
write.table(x, file = "v_and_h_Geary.txt", sep = "\t", row.names = FALSE, quote = FALSE)

# 提取Moran
x <- sapply(pos, extractMoran)
# 保存txt
write.table(x, file = "v_and_h_Moran.txt", sep = "\t", row.names = FALSE, quote = FALSE)

# 提取MoreauBroto
x <- sapply(pos, extractMoreauBroto)
# 保存txt
write.table(x, file = "v_and_h_MoreauBroto.txt", sep = "\t", row.names = FALSE, quote = FALSE)

# 提取PAAC
x <- sapply(pos, extractPAAC)
# 保存txt
write.table(x, file = "v_and_h_PAAC.txt", sep = "\t", row.names = FALSE, quote = FALSE)

# 提取QSO
x <- sapply(pos, extractQSO)
# 保存txt
write.table(x, file = "v_and_h_QSO.txt", sep = "\t", row.names = FALSE, quote = FALSE)

# 提取SOCN
x <- sapply(pos, extractSOCN)
# 保存txt
write.table(x, file = "v_and_h_SOCN.txt", sep = "\t", row.names = FALSE, quote = FALSE)

# 提取TC
x <- sapply(pos, extractTC)
# 保存txt
write.table(x, file = "v_and_h_TC.txt", sep = "\t", row.names = FALSE, quote = FALSE)

# 提取PSSMFeature
rm(list = ls())
setwd("pssm")
folder_path <- "."
file_names <- tools::file_path_sans_ext(list.files(folder_path))
result_df <- data.frame(matrix(nrow = 1200, ncol = 0))
for (file_name in file_names) {
  print(file_name)
  # 读取 PSSM 文件
  pssmdf <- t(read.table(file.path(folder_path, paste0(file_name, ".pssm")), sep = " "))
  pssmmat <- as.matrix(pssmdf)
  pssmac <- as.vector(extractPSSMAcc(pssmmat, lag = 3))
  result_df[[file_name]] <- pssmac
}
write.table(result_df, file = "../v_and_h_PSSMAC.txt", sep = "\t", row.names = FALSE, quote = FALSE)
