import re
fout = open("bert-large-uncased.30522.1024dnew.vec", "w", encoding='UTF-8')
with open("bert-large-uncased.30522.1024d.vec", "r", encoding='UTF-8') as fin:
    header = fin.readline()
    fout.write(header)
    vocab_size, vector_size = (int(x) for x in header.split())  # throws for invalid file format

    for line_no in range(vocab_size):
        line = fin.readline()
        #这边写个正则表达式匹配两个空格的换成一个空格即可
        line = re.sub("[ ]+", " ", line)
        fout.write(line)
fout.close()
print("Finished")