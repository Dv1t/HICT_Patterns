import numpy as np
from Bio import SeqIO
import gzip
import argparse

def make_reverse(seq, end):
    length = np.random.random_integers(200000, 10000000)
    start = end - length
    if start <=0:
        return False
    seq = seq[:start] + seq[start:end].reverse_complement() + seq[end:]

    return seq, (start, end), start

def make_translocation(seq, end, reverse=False):
    length = np.random.random_integers(200000, 1000000)
    start = end - length

    if start <=0:
        return False


    move_length = np.random.random_integers(200000, 2000000)
    moving_start = start + np.random.random_integers(200000, 2000000)
    move_seq = seq[moving_start:moving_start+move_length]

    if reverse:
        move_seq = move_seq.reverse_complement()

    move_point = moving_start + move_length + (moving_start-start)  + move_length
    seq = seq[:moving_start] + seq[moving_start+move_length:move_point] + move_seq + seq[move_point:]

    return seq, (moving_start, move_point-move_length), (moving_start, move_point), (move_point-move_length, move_point), start

def generate(fasta_path, target_sv_number, label):
    records = {}
    if '.gz' in fasta_path:
        with gzip.open(fasta_path, "rt") as handle:
            for record in (SeqIO.parse(handle, "fasta")):
                if len(record.seq) > 10000000:
                    records[record.id] = record
    else:
        with open(fasta_path) as handle:
            for record in (SeqIO.parse(handle, "fasta")):
                if len(record.seq) > 10000000:
                    records[record.id] = record

    sv_count = 0
    file_number = 1
    while sv_count<target_sv_number:

        seqs = {}
        with open(f'{label}_sv_{file_number}.csv', mode='w') as output:
            output.write(f'chr,label,start,end\n')
            for chr in records.keys():
                seq = records[chr].seq
                end = len(seq)
                while end > 10000000:
                    seq, coords, end = make_reverse(seq, end)
                    output.write(f'{chr},inversion,{coords[0]},{coords[1]}\n')
                    if end < 5000000:
                        seqs[chr] = seq
                        continue
                    seq, plus_minus, minus_plus_1, plus_plus_2, end = make_translocation(seq, end)
                    output.write(f'{chr},translocation-,{plus_minus[0]},{plus_minus[1]}\n')
                    output.write(f'{chr},translocation+,{minus_plus_1[0]},{minus_plus_1[1]}\n')
                    output.write(f'{chr},translocation,{plus_plus_2[0]},{plus_plus_2[1]}\n')

                    end -= 1000000
                    if end < 5000000:
                        seqs[chr] = seq
                        continue
                    seq, plus_minus, minus_plus_1, minus_plus_2, end = make_translocation(seq, end, True)
                    output.write(f'{chr},translocation_reversed-,{plus_minus[0]},{plus_minus[1]}\n')
                    output.write(f'{chr},translocation_reversed+, {minus_plus_1[0]},{minus_plus_1[1]}\n')
                    output.write(f'{chr},translocation_reversed,{minus_plus_2[0]},{minus_plus_2[1]}\n')
                    sv_count+=7
                seqs[chr] = seq
    
        out_fasta_path = f"{label}_with_sv_{file_number}.fasta.gz"
        with gzip.open(out_fasta_path, 'wt', encoding='utf-8') as output:
            for chr in records.keys():
                output.write(f'>{chr}\n')
                output.write(str(seqs[chr]))
                output.write('\n')
        file_number+=1


parser = argparse.ArgumentParser()
parser.add_argument('fasta_path', type=str, help='Path to fasta for SVs generation')
parser.add_argument('target_sv_number', type=int, help='Number of SVs needed, files will be generated until this number is reached')
parser.add_argument('label', type=str, help='Common name for generated files - FASTAs and .csv with SVs')

args = parser.parse_args()

generate(args.fasta_path, int(args.target_sv_number), args.label)