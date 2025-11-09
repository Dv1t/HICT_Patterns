SRA_ID="SRR25223524"

echo "prefetch -X 9999999999999 $SRA_ID -O ./"
prefetch -X 9999999999999 $SRA_ID -O ./
echo

echo "fasterq-dump ./${SRA_ID} -O ./"
fasterq-dump fasterq-dump ./${SRA_ID} -O ./
echo

rm -r ./${SRA_ID}

echo "gzip ./${SRA_ID}_1.fastq"
gzip ./${SRA_ID}_1.fastq
echo
echo "gzip ./${SRA_ID}_2.fastq"
gzip ./${SRA_ID}_2.fastq
echo


READS1="./${SRA_ID}_1.fastq.gz"
READS2="./${SRA_ID}_2.fastq.gz"

# picard - to get the picard.jar, please use wget https://github.com/broadinstitute/picard/releases/download/3.0.0/picard.jar
PICARD=picard/picard.jar

#path to jdk version >=17
export JAVA_HOME="java/jdk-20.0.1"
export PATH=$JAVA_HOME/bin:$PATH

#source /nfs/home/vdravgelis/miniforge3/bin/activate
#conda activate arima

# path to pairix and bam2pairs (requiers installed pairix: https://github.com/4dn-dcic/pairix#installation-for-pairix)
PATH=~/pairix/bin/:~/pairix/util:~/pairix/util/bam2pairs:$PATH       
                                                                     
# number of threads
THREADS=32

REF="chm_transformed_2.fasta.gz"

LABEL=Siamang_SV_2

RAW_DIR=${LABEL}/RAW
FILT_DIR=${LABEL}/FILTER
COMB_DIR=${LABEL}/COMB

mkdir -p $RAW_DIR
mkdir -p $FILT_DIR
mkdir -p $COMB_DIR

echo "bwa index $REF"
bwa index $REF
echo

echo "samtools faidx $REF"
samtools faidx $REF
echo

echo "### Step 1.A: FASTQ to BAM (1st)"
echo "\
bwa mem -t $THREADS $REF $READS1 | samtools view -Sb - > $RAW_DIR/${LABEL}_1.bam"
bwa mem -t $THREADS $REF $READS1 | samtools view -Sb - > $RAW_DIR/${LABEL}_1.bam
echo ""

echo "### Step 2.A: Filter 5' end (1st)"
echo "\
samtools view -h ${RAW_DIR}/${LABEL}_1.bam | perl filter_5end.pl | samtools view -@ $THREADS -Sb - > $FILT_DIR/${LABEL}_1.bam"
samtools view -h ${RAW_DIR}/${LABEL}_1.bam | perl filter_5end.pl | samtools view -@ $THREADS -Sb - > $FILT_DIR/${LABEL}_1.bam &&
echo "" || exit -1

echo "### Remove intermediate file"
echo "\
rm $RAW_DIR/${LABEL}_1.bam"
rm $RAW_DIR/${LABEL}_1.bam
echo

echo "### Step 1.B: FASTQ to BAM (2nd)"
echo "\
bwa mem -t $THREADS $REF $READS2 | samtools view -Sb - > $RAW_DIR/${LABEL}_2.bam"
bwa mem -t $THREADS $REF $READS2 | samtools view -Sb - > $RAW_DIR/${LABEL}_2.bam
echo ""

echo "### Step 2.B: Filter 5' end (2nd)"
echo "\
samtools view -h ${RAW_DIR}/${LABEL}_2.bam | perl filter_5end.pl | samtools view -@ $THREADS -Sb - > $FILT_DIR/${LABEL}_2.bam"
samtools view -h ${RAW_DIR}/${LABEL}_2.bam | perl filter_5end.pl | samtools view -@ $THREADS -Sb - > $FILT_DIR/${LABEL}_2.bam &&
echo "" || exit -1

echo "### Remove intermediate file"
echo "\
rm $RAW_DIR/${LABEL}_2.bam"
rm $RAW_DIR/${LABEL}_2.bam
echo


echo "### Step 3.A: Filter Combiner"
echo "\
perl two_read_bam_combiner.pl $FILT_DIR/${LABEL}_1.bam $FILT_DIR/${LABEL}_2.bam | samtools view -@ $THREADS -Sb > $COMB_DIR/${LABEL}.bam"
perl two_read_bam_combiner.pl $FILT_DIR/${LABEL}_1.bam $FILT_DIR/${LABEL}_2.bam | samtools view -@ $THREADS -Sb > $COMB_DIR/${LABEL}.bam
echo ""

echo "mv $COMB_DIR/${LABEL}.bam ${LABEL}/"
mv $COMB_DIR/${LABEL}.bam ${LABEL}/

echo "#### Finished Mapping!"
echo ""

echo "### Start to dedup"

bam=${LABEL}/${LABEL}.bam

mkdir -p ${LABEL}/sort_dedup
bam_prefix=`basename $bam`
sort_bam=${LABEL}/sort_dedup/${bam_prefix/.bam/.sort.bam}
dp_bam=${sort_bam/.bam/.dp.bam}
resort_bam=${dp_bam/.bam/.sort_n.bam}

echo "# Sorting by coordinates"
echo "\
samtools sort -@ $THREADS -T $sort_bam.tmp -m2G -O bam -o $sort_bam $bam"
samtools sort -@ $THREADS -T $sort_bam.tmp -m2G -O bam -o $sort_bam $bam
echo "\
samtools index $sort_bam"
samtools index $sort_bam
echo ""

mkdir -p ${LABEL}/tmpS
echo "\
java -jar -Xmx32g -Djava.io.tmpdir=$PWD/${LABEL}/tmp $PICARD MarkDuplicates -REMOVE_DUPLICATES true -I $sort_bam -O $dp_bam -M ${dp_bam/.bam/.metrics.txt} -ASSUME_SORT_ORDER coordinate -MAX_FILE_HANDLES_FOR_READ_ENDS_MAP 1024"
java -jar -Xmx32g -Djava.io.tmpdir=$PWD/${LABEL}/tmp $PICARD MarkDuplicates -REMOVE_DUPLICATES true -I $sort_bam -O $dp_bam -M ${dp_bam/.bam/.metrics.txt} -ASSUME_SORT_ORDER coordinate -MAX_FILE_HANDLES_FOR_READ_ENDS_MAP 1024
echo ""

echo "# Resort to name order"
echo "\
samtools sort -@ $THREADS -n -T $resort_bam.tmp -m2G -O bam -o $resort_bam $dp_bam"
samtools sort -@ $THREADS -n -T $resort_bam.tmp -m2G -O bam -o $resort_bam $dp_bam
echo ""

echo "# Counting statistics"
echo "\
perl get_stats.pl $resort_bam > $LABEL/dedups_bam.stats"
perl get_stats.pl $resort_bam > $LABEL/dedups_bam.stats
echo ""

echo "### Step 4: Convert to cooler"

BAM="${LABEL}/sort_dedup/${LABEL}.sort.dp.sort_n.bam"

TMPDIR_NAME="TEMP_${LABEL}"
mkdir -p $TMPDIR_NAME
export TMPDIR=$TMPDIR_NAME
mkdir -p $LABEL

echo "\
python create_chrom_sizes.py $REF $LABEL"
python create_chrom_sizes.py $REF $LABEL

echo "\
bam2pairs -c ${LABEL}/chrom.sizes $BAM ${LABEL}/${LABEL}"
bam2pairs -c ${LABEL}/chrom.sizes $BAM ${LABEL}/${LABEL}

echo "\
python -m cooler cload pairix -p $CPU ${LABEL}/chrom.sizes:1000 ${LABEL}/${LABEL}.bsorted.pairs.gz ${LABEL}/${LABEL}_1k.cool"
python -m cooler cload pairix -p $CPU ${LABEL}/chrom.sizes:1000 ${LABEL}/${LABEL}.bsorted.pairs.gz ${LABEL}/${LABEL}_1k.cool

echo "\
python -m cooler zoomify -n $CPU -r 4DN --balance --balance-args '--nproc 32' -o ${LABEL}/${LABEL}_4DN.mcool ${LABEL}/${LABEL}_1k.cool"
python -m cooler zoomify -n $CPU -r 4DN --balance --balance-args '--nproc 32' -o ${LABEL}/${LABEL}_4DN.mcool ${LABEL}/${LABEL}_1k.cool

echo "\
Job completed!"