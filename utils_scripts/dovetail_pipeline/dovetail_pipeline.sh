#!/bin/bash

# this will mark data directory - this is very imortant to write - this is directory of the stript and all files needed for run
#SBATCH -D /mnt/tank/scratch/vdravgelis/ClusterBuffer

# this will mark the amount of total memory
#SBATCH --mem=256G

# number of cpu
#SBATCH --cpus-per-task=32
CPU=32
#time in hours:mins:seconds
#SBATCH --time=72:00:00

# out and err file (change the name): %j is number of procces running, out file just marking the running command, err file is crutial for understanding what went wrong
#SBATCH --output=make_map.%j.%N.txt
#SBATCH --error=make_map.%j.%N.err
#SBATCH --mail-type=END,FAIL

# this will notify you about the run status
#SBATCH --mail-user=vitdrav@gmail.com

SRA_ID="SRR25222170"

source /nfs/home/vdravgelis/miniforge3/bin/activate
export PATH=/nfs/home/vdravgelis/miniforge3/bin/:$PATH
. /nfs/home/vdravgelis/miniforge3/etc/profile.d/mamba.sh

#mamba activate parallel-fastq-dump
#parallel-fastq-dump -s ${SRA_ID} -t 16  --split-files --gzip

READS1="/mnt/tank/scratch/vdravgelis/data/hic/apes/${SRA_ID}_1.fastq.gz"
READS2="/mnt/tank/scratch/vdravgelis/data/hic/apes/${SRA_ID}_2.fastq.gz"

# picard - to get the picard.jar, please use wget https://github.com/broadinstitute/picard/releases/download/3.0.0/picard.jar
PICARD=picard/picard.jar

#path to jdk version >=17
export JAVA_HOME="java/jdk-20.0.1"
export PATH=$JAVA_HOME/bin:$PATH

mamba activate dovetail


# path to pairix and bam2pairs (requiers installed pairix: https://github.com/4dn-dcic/pairix#installation-for-pairix)
PATH=~/pairix/bin/:~/pairix/util:~/pairix/util/bam2pairs:$PATH       
                                                                     
# number of threads
THREADS=32

REF="/mnt/tank/scratch/vdravgelis/data/genomes/apes/gor_transformed.fasta.gz"

LABEL=Gor_SV
RAW_DIR=${LABEL}/RAW
FILT_DIR=${LABEL}/FILTER
TEMP_DIR=${LABEL}/TEMP

mkdir -p $LABEL
mkdir -p $RAW_DIR
mkdir -p $FILT_DIR
mkdir -p $TEMP_DIR

echo "bwa index $REF"
bwa index $REF
echo

echo "samtools faidx $REF"
samtools faidx  $REF
echo

echo "
cut -f1,2 ${REF}.fai > $LABEL/${LABEL}.genome
"
cut -f1,2 ${REF}.fai > $LABEL/${LABEL}.genome
echo

GENOME=$LABEL/${LABEL}.genome

echo "### Step 1: FASTQ to BAM"
echo "\
bwa mem -5SP -T0 -t $THREADS $REF $READS1 $READS2 | bgzip > $RAW_DIR/aligned.sam.gz
"
bwa mem -5SP -T0 -t $THREADS $REF $READS1 $READS2 | bgzip > $RAW_DIR/aligned.sam.gz
echo ""

echo "\
pairtools parse --min-mapq 40 --walks-policy 5unique \
--max-inter-align-gap 30 --nproc-in $THREADS --nproc-out $THREADS --chroms-path $GENOME $RAW_DIR/aligned.sam.gz | \
pairtools sort --tmpdir=${TEMP_DIR} --nproc $THREADS | pairtools dedup --nproc-in $THREADS \
--nproc-out $THREADS --mark-dups --output-stats ${LABEL}/stats.txt | pairtools split --nproc-in $THREADS \
--nproc-out $THREADS --output-pairs $FILT_DIR/${LABEL}_mapped.pairs --output-sam - | samtools view -bS -@ ${THREADS} | \
samtools sort -@ ${THREADS} -o $FILT_DIR/${LABEL}_mapped.PT.bam
"
pairtools parse --min-mapq 40 --walks-policy 5unique \
--max-inter-align-gap 30 --nproc-in $THREADS --nproc-out $THREADS --chroms-path $GENOME $RAW_DIR/aligned.sam.gz | \
pairtools sort --tmpdir=${TEMP_DIR} --nproc $THREADS | pairtools dedup --nproc-in $THREADS \
--nproc-out $THREADS --mark-dups --output-stats ${LABEL}/stats.txt | pairtools split --nproc-in $THREADS \
--nproc-out $THREADS --output-pairs $FILT_DIR/${LABEL}_mapped.pairs #--output-sam - | samtools view -bS -@ ${THREADS} | \
#samtools sort -@ ${THREADS} -o $FILT_DIR/${LABEL}_mapped.PT.bam
echo ""

#echo "\
#samtools index $FILT_DIR/${LABEL}_mapped.PT.bam
#"
#samtools index $FILT_DIR/${LABEL}_mapped.PT.bam
#echo ""

echo "\
bgzip $FILT_DIR/${LABEL}_mapped.pairs
"
bgzip $FILT_DIR/${LABEL}_mapped.pairs
echo ""

echo "### Step 2: Convert to cooler"

PAIRS="$FILT_DIR/${LABEL}_mapped.pairs.gz"

echo "\
pairix $PAIRS
"
pairix $PAIRS
echo ""

mamba deactivate
mamba activate arima

echo "\
python -m cooler cload pairix -p $THREADS $LABEL/${LABEL}.genome:1000 ${PAIRS} ${LABEL}/${LABEL}_1k.cool
"
python -m cooler cload pairix -p $THREADS $LABEL/${LABEL}.genome:1000 ${PAIRS} ${LABEL}/${LABEL}_1k.cool

echo "\
python -m cooler zoomify -n $CPU -r 1000,5000,15000,25000,50000 --balance --balance-args '--nproc 32' -o ${LABEL}/${LABEL}.mcool ${LABEL}/${LABEL}_1k.cool"
python -m cooler zoomify -n $CPU -r 1000,5000,15000,25000,50000 --balance --balance-args '--nproc 32' -o ${LABEL}/${LABEL}.mcool ${LABEL}/${LABEL}_1k.cool

echo "\
Job completed!"