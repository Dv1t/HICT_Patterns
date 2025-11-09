from Bio import SeqIO
import gzip
import sys

ref_fasta, cur_dir = sys.argv[1:3]

with open(f'{cur_dir}/chrom.sizes', 'w') as outfile:
	if (str(ref_fasta).endswith(".gz") or str(ref_fasta).endswith(".gzip")):
 		with gzip.open(ref_fasta, mode="rt") as uncompressed_fasta:
    			for rec in SeqIO.parse(uncompressed_fasta, 'fasta'):
        			print(f"{rec.id}\t{len(rec)}", file=outfile)

	else:
    		for rec in SeqIO.parse(ref_fasta, 'fasta'):
        		print(f"{rec.id}\t{len(rec)}", file=outfile)


