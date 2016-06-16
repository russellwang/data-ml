//package com.dp.ml.data;
//
///**
// *
// */
//import org.apache.hadoop.fs.Path;
//import org.apache.hadoop.mapred.JobConf;
//import org.apache.mahout.clustering.conversion.InputDriver;
//import org.apache.mahout.clustering.kmeans.KMeansDriver;
//import org.apache.mahout.clustering.kmeans.RandomSeedGenerator;
//import org.apache.mahout.common.distance.DistanceMeasure;
//import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
//import org.apache.mahout.utils.clustering.ClusterDumper;
//import org.conan.mymahout.hdfs.HdfsDAO;
//import org.conan.mymahout.recommendation.ItemCFHadoop;
//
//public class KmeansHadoop {
//    private static final String HDFS = "hdfs://192.168.1.210:9000";
//
//    public static void main(String[] args) throws Exception {
//        String localFile = "datafile/randomData.csv";
//        String inPath = HDFS + "/user/hdfs/mix_data";
//        String seqFile = inPath + "/seqfile";
//        String seeds = inPath + "/seeds";
//        String outPath = inPath + "/result/";
//        String clusteredPoints = outPath + "/clusteredPoints";
//
//        JobConf conf = config();
//        HdfsDAO hdfs = new HdfsDAO(HDFS, conf);
//        hdfs.rmr(inPath);
//        hdfs.mkdirs(inPath);
//        hdfs.copyFile(localFile, inPath);
//        hdfs.ls(inPath);
//
//        InputDriver.runJob(new Path(inPath), new Path(seqFile), "org.apache.mahout.math.RandomAccessSparseVector");
//
//        int k = 3;
//        Path seqFilePath = new Path(seqFile);
//        Path clustersSeeds = new Path(seeds);
//        DistanceMeasure measure = new EuclideanDistanceMeasure();
//        clustersSeeds = RandomSeedGenerator.buildRandom(conf, seqFilePath, clustersSeeds, k, measure);
//        KMeansDriver.run(conf, seqFilePath, clustersSeeds, new Path(outPath), measure, 0.01, 10, true, 0.01, false);
//
//        Path outGlobPath = new Path(outPath, "clusters-*-final");
//        Path clusteredPointsPath = new Path(clusteredPoints);
//        System.out.printf("Dumping out clusters from clusters: %s and clusteredPoints: %s\n", outGlobPath, clusteredPointsPath);
//
//        ClusterDumper clusterDumper = new ClusterDumper(outGlobPath, clusteredPointsPath);
//        clusterDumper.printClusters(null);
//    }
//    
//    public static JobConf config() {
//        JobConf conf = new JobConf(ItemCFHadoop.class);
//        conf.setJobName("ItemCFHadoop");
//        conf.addResource("classpath:/hadoop/core-site.xml");
//        conf.addResource("classpath:/hadoop/hdfs-site.xml");
//        conf.addResource("classpath:/hadoop/mapred-site.xml");
//        return conf;
//    }
//
//}