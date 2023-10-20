import sacrebleu
import sys
def bleu(pred, ref, out_file, tgt):
    with open(pred, 'r', encoding='utf-8') as f:
        pred = f.readlines()

    with open(ref, 'r', encoding='utf-8') as f:
        ref = f.readlines()
        if tgt == 'zh' or tgt == 'cn':
            ours_bleu = sacrebleu.corpus_bleu(pred, [ref], tokenize="zh")
        else:
            ours_bleu = sacrebleu.corpus_bleu(pred, [ref])
    log_message = "BLEU score = {0:.3f}, BP = {1:.3f}, sys_len = {2:}, ref_len = {3:}\n".format(ours_bleu.score, ours_bleu.bp, ours_bleu.sys_len, ours_bleu.ref_len)
    print(log_message, end="") 
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(log_message)

def main():
    # pred, ref, output_file, to_language
    bleu(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

if __name__ == "__main__":
    main()
