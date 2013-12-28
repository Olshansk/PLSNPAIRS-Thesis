package npairs;

import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import java.io.*;
import java.util.*;
import java.net.*;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.*;
import org.apache.hadoop.util.*;

import npairs.NpairsjSetupParams;
import npairs.io.NpairsDataLoader;
import npairs.shared.matlib.Matrix;
import npairs.shared.matlib.MatrixImpl;
import npairs.utils.CVA;
import npairs.utils.PCA;
import npairs.utils.ZScorePatternInfo;
import npairs.utils.PredictionStats;

public class Test {

	public static Matrix[] ppTrueClass;      
	public static Matrix[] sqrdPredError;    
	public static Matrix[] predClass;         
	public static Matrix[] correctPred;       
	public static Matrix[][] ppAllClasses;  
	private static Matrix[] r2; 
	
	public static String hadoopDirectory = "/user/" + System.getProperty("user.name") + "/";
	static String localDirectory = System.getProperty("user.dir" );

	//ALan: note: it used to be non-static
	private static Matrix getTrainCVScores(CVA cvaTrain, int[] trainDataVols, NpairsjSetupParams setupParams) {
		Matrix cvsTr = cvaTrain.getCVScores();
		Matrix cvsTrPadded = cvsTr.zeroPadRows(setupParams.numVols, trainDataVols);

		return cvsTrPadded;
	}
	private static Matrix getTestCVScores(CVA cvaTrain, int[] testDataVols, NpairsjSetupParams setupParams, Matrix FeatSelData) {
		Matrix testData = null;	

		if (setupParams.initFeatSelect) {
			testData = FeatSelData.subMatrixRows(testDataVols);
		}
		/*
		else {
			testData = dataLoader.getOrigData().subMatrixRows(testDataVols);
		}
		*/
		testData = testData.meanCentreColumns();
		
		Matrix cvsTest = cvaTrain.calcTestCVScores(testData);
		Matrix cvsTestPadded = cvsTest.zeroPadRows(setupParams.numVols, testDataVols);
		return cvsTestPadded;
	}
	private static void initPredStats(int totalNoSplitAnalyses) {
		ppTrueClass = new Matrix[totalNoSplitAnalyses];
		sqrdPredError = new Matrix[totalNoSplitAnalyses];
		predClass = new Matrix[totalNoSplitAnalyses];
		correctPred = new Matrix[totalNoSplitAnalyses];
		ppAllClasses = new Matrix[totalNoSplitAnalyses][2];
	}

	private static void computePredStats(int numAnalyses, 
			int[] split1DataVols, 
			int[] split2DataVols, 
			Matrix split1CVSTrain, 
			Matrix split1CVSTest, 
			Matrix split2CVSTrain, 
			Matrix split2CVSTest,NpairsjSetupParams setupParams) throws NpairsjException {

		PredictionStats predStats1 = new PredictionStats(split1CVSTrain, split2CVSTest,
				split1DataVols, split2DataVols, setupParams.getClassLabels());

		// add to cumulative prediction stats
		int currAnalysis = numAnalyses;
		if (setupParams.switchTrainAndTestSets) {
			currAnalysis--;  // numAnalyses was incremented twice but curr 
			// analysis is for first train set
		}
		ppTrueClass[currAnalysis - 1] = new MatrixImpl(predStats1.getPPTrueClass()).getMatrix();
		sqrdPredError[currAnalysis - 1] = new MatrixImpl(predStats1.getSqrdPredError()).getMatrix();
		predClass[currAnalysis - 1] = new MatrixImpl(predStats1.getPredClass()).getMatrix();
		correctPred[currAnalysis - 1] = new MatrixImpl(predStats1.getCorrectPred()).getMatrix();
		ppAllClasses[currAnalysis - 1][0] = new MatrixImpl(predStats1.getPPAllClasses(0)).getMatrix();
		ppAllClasses[currAnalysis - 1][1] = new MatrixImpl(predStats1.getPPAllClasses(1)).getMatrix();

		PredictionStats predStats2 = null;
		if (setupParams.switchTrainAndTestSets) {
			currAnalysis++; // was decremented for first train set

			predStats2 = new PredictionStats(split2CVSTrain, split1CVSTest,split2DataVols, split1DataVols, setupParams.getClassLabels());			

			ppTrueClass[currAnalysis - 1] = new MatrixImpl(predStats2.getPPTrueClass()).getMatrix();
			sqrdPredError[currAnalysis - 1] = new MatrixImpl(predStats2.getSqrdPredError()).getMatrix();
			predClass[currAnalysis - 1] = new MatrixImpl(predStats2.getPredClass()).getMatrix();
			correctPred[currAnalysis - 1] = new MatrixImpl(predStats2.getCorrectPred()).getMatrix();
			ppAllClasses[currAnalysis - 1][0] = new MatrixImpl(predStats2.getPPAllClasses(0)).getMatrix();
			ppAllClasses[currAnalysis - 1][1] = new MatrixImpl(predStats2.getPPAllClasses(1)).getMatrix();
		}
	}
	private static void addCurrR2(int splitNum, NpairsjSetupParams setupParams, CVA splitDataCVA1, CVA splitDataCVA2) {
		int analysisNum = splitNum;
		if (setupParams.switchTrainAndTestSets) {
			analysisNum *= 2;
		}
		if (setupParams.cvaRun) {
			r2[analysisNum] = splitDataCVA1.getR2();
			if (setupParams.switchTrainAndTestSets) {
				r2[analysisNum + 1] = splitDataCVA2.getR2();
			}	
		}	
	}
	private static void savePCASplitResults(int splitNum, int splitHalf,NpairsjSetupParams setupParams, PCA splitDataPCA1, PCA splitDataPCA2) throws NpairsjException { 

		String pcaSavePref = setupParams.resultsFilePrefix;
		if (!setupParams.pcEigimsToBigSpace) {
			pcaSavePref += ".InitFSpace";
		}
		if (splitHalf == 1) {
			splitDataPCA1.savePCAResultsIDL(pcaSavePref, null, true,
					splitNum, splitHalf);
			// save denoised (i.e. PCA dim-reduced) input data 
			// (in orig img space) 
			// TODO: determine if one would ever want split image
			// data saved (doesn't seem likely)
//			if (setupParams.saveDataPostPCA) {
//				splitDataPCA1.saveDataPostPCA(dataLoader);
//			}
		}
		else {
			splitDataPCA2.savePCAResultsIDL(pcaSavePref, null, true,
					splitNum, splitHalf);
			// save denoised (i.e. PCA dim-reduced) input data 
			// (in orig img space)
//			if (setupParams.saveDataPostPCA) {
//				splitDataPCA2.saveDataPostPCA(dataLoader);
//			}
		}
	}
  public static class TokenizerMapper 
       extends Mapper<Object, Text, Text, IntWritable>{
    
    //private final static IntWritable one = new IntWritable(1);
   // private Text word = new Text();
      
    public void map(Object key, Text value, Context context
                    ) throws IOException, InterruptedException {    	
    	
    	double sTime1 = System.currentTimeMillis();
    	
        /*****************Reconstructing the dataloader object**************/        
        FileSystem fs1 = FileSystem.get(context.getConfiguration());
        Path path = new Path("dataLoader.ser");
      
        InputStream in = fs1.open(path);
        ObjectInputStream objReader = new ObjectInputStream(in);
        NpairsDataLoader dataLoader = null;
                  
  		try {
  			dataLoader = (NpairsDataLoader)objReader.readObject();
  		} catch (ClassNotFoundException e) {
  			// TODO Auto-generated catch block
  			e.printStackTrace();
  		}
        in.close();  
        /******************Reconstructing the setupParams object**************/
        FileSystem fs2 = FileSystem.get(new Configuration());
        Path path2 = new Path("setupParams.ser");
        InputStream in2 = fs2.open(path2);
        ObjectInputStream objReader2 = new ObjectInputStream(in2);
        //fs2.deleteOnExit(path2);
        NpairsjSetupParams setupParams = null;
        try {
        	setupParams = (NpairsjSetupParams)objReader2.readObject();
        } catch (ClassNotFoundException e) {
  		// TODO Auto-generated catch block
  		e.printStackTrace();
        }
        in2.close();           
        /******************Reconstructing the splits object**************/
        FileSystem fs3 = FileSystem.get(new Configuration());
        Path path3 = new Path("splits.ser");
        InputStream in3 = fs3.open(path3);
        ObjectInputStream objReader3 = new ObjectInputStream(in3);
        int[][][] splits = null;
        try {
        	splits = (int[][][])objReader3.readObject();
        } catch (ClassNotFoundException e) {
  		// TODO Auto-generated catch block
  		e.printStackTrace();
        }
        in3.close();                
        /******************Reconstructing the fullDataAnalysis object**************/
        //FileSystem fs4 = FileSystem.get(new Configuration());
        Path path4 = new Path("fullDataAnalysis.ser");
        InputStream in4 = fs3.open(path4);
        ObjectInputStream objReader4 = new ObjectInputStream(in4);
        Analysis fullDataAnalysis = null;    
        try {
        	fullDataAnalysis = (Analysis)objReader4.readObject();
        } catch (ClassNotFoundException e) {
  		// TODO Auto-generated catch block
  		e.printStackTrace();
        }
        in4.close();           
        /*************************************************************/
    	
        double tTime1 = (System.currentTimeMillis() - sTime1) / 1000;
		System.out.println("De-Serialize required objects takes [" + tTime1 + " s]");
        
        String[] splitNums = value.toString().trim().split("_");
		int numSamples = 100;

        int nCVDimsSplit1 = setupParams.cvaPCSet1.length;
		int nCVDims = Math.min(fullDataAnalysis.getCVA().getNumCVDims(), nCVDimsSplit1);
		Matrix split1CVAEvals = new MatrixImpl(numSamples, nCVDims).getMatrix();  
		double[] avgSplit1CVAEvals = new double[nCVDims];
		Matrix avgCVScoresTrain = new MatrixImpl(setupParams.numVols, nCVDims).getMatrix();
		Matrix avgCVScoresTest = new MatrixImpl(setupParams.numVols, nCVDims).getMatrix();
		Matrix split2CVAEvals = new MatrixImpl(100, nCVDims).getMatrix();

		Matrix avgSpatialPattern = null;
		Matrix avgZScorePattern = null;
		Matrix avgNoisePattern = null;
		Matrix corrCoeffs = null;       
		Matrix noisePattStdDev = null; 
		int count = 0;
		int numAnalyses = 0;
		
		int totalNumSplitAnalyses = numSamples;
		if (setupParams.switchTrainAndTestSets) {
			totalNumSplitAnalyses = 2 * numSamples;
		}
		initPredStats(totalNumSplitAnalyses);
		r2 = new Matrix[totalNumSplitAnalyses];
		numAnalyses = Integer.parseInt(splitNums[0]);
		if (setupParams.switchTrainAndTestSets) {
			numAnalyses = numAnalyses * 2;
		}
		
        for(String i:splitNums){
            
        	int splitNum = Integer.parseInt(i);

            Analysis firstPartAnalysis = null;
    		Analysis secondPartAnalysis = null;

    		int[] split1DataVols = splits[0][splitNum];
    		int[] split2DataVols = splits[1][splitNum];
    		
    		if (setupParams.initFeatSelect) {
    			firstPartAnalysis = 
    				new Analysis(dataLoader.getFeatSelData(), setupParams, 
    						split1DataVols, true, fullDataAnalysis);
    			if (setupParams.switchTrainAndTestSets) {
    				secondPartAnalysis = 
    					new Analysis(dataLoader.getFeatSelData(), setupParams, 
    							split2DataVols, false, fullDataAnalysis);
    			}
    		}
    		else {
    			firstPartAnalysis = 
    				new Analysis(dataLoader.getOrigData(), setupParams, 
    						split1DataVols, true, fullDataAnalysis);
    			if (setupParams.switchTrainAndTestSets) {
    				secondPartAnalysis =
    					new Analysis(dataLoader.getOrigData(), setupParams, 
    							split2DataVols, false, fullDataAnalysis);
    			}
    		}
    		 
    		try {
    			firstPartAnalysis.run();
    		} catch (NpairsjException e) {
    			// TODO Auto-generated catch block
    			e.printStackTrace();
    		}
    		
    		++numAnalyses;

    		if (setupParams.switchTrainAndTestSets) {

    			try {
    				secondPartAnalysis.run();
    			} catch (NpairsjException e) {
    				// TODO Auto-generated catch block
    				e.printStackTrace();
    			}
    			++numAnalyses;
    		}
    		
    		//Added by Alan
    		PCA splitDataPCA1 = null;
    		PCA splitDataPCA2 = null;
    		CVA splitDataCVA1 = null;
    		CVA splitDataCVA2 = null;
    		
    		if (setupParams.pcaRun) {
    			splitDataPCA1 = firstPartAnalysis.getPCA(); 
    			if (setupParams.switchTrainAndTestSets) {
    				splitDataPCA2 = secondPartAnalysis.getPCA();
    			}
    		}
    		
    		splitDataCVA1 = firstPartAnalysis.getCVA();  // null if cva not run
    		splitDataCVA2 = secondPartAnalysis.getCVA(); // null if cva not run or training and 
    													 // test data not switched

    		// Add r2 stats Matrix for current split half to r2 Matrix array.
			if (true) {
				addCurrR2(splitNum,setupParams,splitDataCVA1,splitDataCVA2);
			}
    		
    		Matrix split1CVSTrain = null;
			Matrix split1CVSTest = null;
			Matrix split2CVSTrain = null;
			Matrix split2CVSTest = null;

			if (setupParams.cvaRun) {
				//  Incorporate current split into summary CV-Training and CV-Test Scores				
				split1CVAEvals.setRow(splitNum, splitDataCVA1.getEvals());

				// Split1CVSTrain = 0 for vols not incl. in current split data
				split1CVSTrain = getTrainCVScores(splitDataCVA1, split1DataVols, setupParams);
				
				/*Alan: skip for now
				for (int vol : split1DataVols) {
					vCountTr[vol] += 1;
				}
				*/

				avgCVScoresTrain = avgCVScoresTrain.plus(split1CVSTrain);

				// Project curr split test (split2) data onto curr split train (split1) 
				// CV eigenimages to get curr test CV scores. 

				// Split2CVSMatrix contains zeros in rows corresp. to vols not
				// incl. in curr test data 
				split2CVSTest = getTestCVScores(splitDataCVA1, split2DataVols, setupParams, dataLoader.getFeatSelData());
						
				/*Alan: skip for now

				for (int vol : split2DataVols) {
					vCountTe[vol] += 1;
				}		
				
				*/
				avgCVScoresTest = avgCVScoresTest.plus(split2CVSTest);

				if (setupParams.switchTrainAndTestSets) {
					split2CVAEvals.setRow(splitNum, splitDataCVA2.getEvals());

					// Split2CVSTrain = 0 for vols not incl. in current split data
					split2CVSTrain = getTrainCVScores(splitDataCVA2, split2DataVols, setupParams);

					/*Alan: skip for now

					for (int vol : split2DataVols) {
						vCountTr[vol] += 1;
					}
					*/
					
					avgCVScoresTrain = avgCVScoresTrain.plus(split2CVSTrain);
					// Project curr split test (split1) data onto curr split train (split2) 
					// CV eigenimages to get curr test CV scores. 

					// Split2CVSMatrix contains zeros in rows corresp. to vols not
					// incl. in curr test data 
					split1CVSTest = getTestCVScores(splitDataCVA2, split1DataVols,  setupParams, dataLoader.getFeatSelData());
					
					/*Alan: skip for now
					for (int vol : split1DataVols) {
						vCountTe[vol] += 1;
					}	
					*/
					
					avgCVScoresTest = avgCVScoresTest.plus(split1CVSTest);
				}
			}
			if (setupParams.initFeatSelect) {
				if (setupParams.cvaRun) {
					splitDataCVA1.rotateEigimsToOrigSpace(dataLoader.getEVDProjFactorMat(), 
							dataLoader.getOrigData());
				}

				if (setupParams.pcaRun) {
					if (setupParams.pcEigimsToBigSpace && setupParams.saveSplitDataResults) {
						splitDataPCA1.rotateEigimsToOrigSpace(setupParams.cvaPCSet1, 
								dataLoader.getEVDProjFactorMat(), dataLoader.getOrigData());
					}
				}

				if (setupParams.switchTrainAndTestSets) {
					if (setupParams.cvaRun) {
						splitDataCVA2.rotateEigimsToOrigSpace(dataLoader.getEVDProjFactorMat(), 
								dataLoader.getOrigData());
					}

					if (setupParams.pcaRun) {
						if (setupParams.pcEigimsToBigSpace && setupParams.saveSplitDataResults) {
							splitDataPCA2.rotateEigimsToOrigSpace(setupParams.cvaPCSet2, 
									dataLoader.getEVDProjFactorMat(), dataLoader.getOrigData());
						}
					}
				}
			}

//			Save split results in ASCII format:
			if (setupParams.saveSplitDataResults) {
				if (setupParams.pcaRun) {
					try {
						savePCASplitResults(splitNum, 1, setupParams,splitDataPCA1,splitDataPCA2);
					} catch (NpairsjException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}

				if (setupParams.cvaRun) {
					splitDataCVA1.saveCVAResultsIDL(setupParams.resultsFilePrefix, 
							true, splitNum, 1, false);
				}

				if (setupParams.switchTrainAndTestSets) {
					if (setupParams.pcaRun) {
						try {
							savePCASplitResults(splitNum, 2,setupParams,splitDataPCA1,splitDataPCA2);
						} catch (NpairsjException e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
						}
					}
					if (setupParams.cvaRun) {
						splitDataCVA2.saveCVAResultsIDL(setupParams.resultsFilePrefix, 
								true, splitNum, 2, false);
					}
				}
			}

    		
			if (setupParams.switchTrainAndTestSets) {
				
				ZScorePatternInfo zScorePattInfo = new ZScorePatternInfo(splitDataCVA1.getEigimsBig(), 
						splitDataCVA2.getEigimsBig());
				
				//Matrix noisePattStdDev;
				//Matrix corrCoeffs;
				if (count == 0) {
					avgSpatialPattern = splitDataCVA1.getEigimsBig();
					avgSpatialPattern = avgSpatialPattern.plus(splitDataCVA2.getEigimsBig());
					avgZScorePattern = zScorePattInfo.getSignalPattern();
					avgNoisePattern = zScorePattInfo.getNoisePattern();

					// must initialize noisePattStdDev and corrCoeffs 
					// because each col added one at a time
					 noisePattStdDev = new MatrixImpl(numSamples, 
							avgNoisePattern.numCols()).getMatrix();
					 corrCoeffs =  new MatrixImpl(numSamples, 
							avgNoisePattern.numCols()).getMatrix();
				}
				else {
					avgSpatialPattern = avgSpatialPattern.plus(
							splitDataCVA1.getEigimsBig());
					avgSpatialPattern = avgSpatialPattern.plus(
							splitDataCVA2.getEigimsBig());
					avgZScorePattern = avgZScorePattern.plus(
							zScorePattInfo.getSignalPattern());
					avgNoisePattern = avgNoisePattern.plus(
							zScorePattInfo.getNoisePattern());
				}
				noisePattStdDev.setRow(splitNum, zScorePattInfo.getNoiseStdDev());
				corrCoeffs.setRow(splitNum, zScorePattInfo.getCorrCoeffs());

				// TODO: lift the following if statement out of splitNum for loop
				//if (splitNum == numSamples - 1) {
				//	avgSpatialPattern = avgSpatialPattern.mult(1.0 / (double)(numSamples * 2));			
				//	avgZScorePattern = avgZScorePattern.mult(1.0 / (double)numSamples);
				//	avgNoisePattern = avgNoisePattern.mult(1.0 / (double)numSamples);					
				//} 		
				
				
			} // end if switch train/test 


			count ++;
			// Prediction metrics can be calculated whenever there is a test 
			// set (or equivalently in NPAIRS, whenever resampling is done)
			try {
				computePredStats(numAnalyses, split1DataVols, split2DataVols, 
						split1CVSTrain, split1CVSTest, split2CVSTrain, split2CVSTest, setupParams);
			} catch (NpairsjException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
        }  
        
        double sTime = System.currentTimeMillis();	
		/******************Serialize objects*********************/
		FileSystem fs4 = FileSystem.get(context.getConfiguration());
		Path temp1 = new Path(hadoopDirectory + "out_npairsj_ser/", "avgCVScoresTrain_" + value.toString().trim());
		ObjectOutputStream os1 = new ObjectOutputStream(fs4.create(temp1));
		os1.writeObject(avgCVScoresTrain);
		os1.flush();
		os1.reset();
		os1.close();
		Path temp2 = new Path(hadoopDirectory + "out_npairsj_ser/", "avgCVScoresTest_" + value.toString().trim());
		ObjectOutputStream os2 = new ObjectOutputStream(fs4.create(temp2));
		os2.writeObject(avgCVScoresTest);
		os2.flush();
		os2.reset();
		os2.close();
		Path temp3 = new Path(hadoopDirectory + "out_npairsj_ser/", "avgSpatialPattern_" + value.toString().trim());
		ObjectOutputStream os3 = new ObjectOutputStream(fs4.create(temp3));
		os3.writeObject(avgSpatialPattern);
		os3.flush();
		os3.reset();
		os3.close();
		Path temp4 = new Path(hadoopDirectory + "out_npairsj_ser/", "avgZScorePattern_" + value.toString().trim());
		ObjectOutputStream os4 = new ObjectOutputStream(fs4.create(temp4));
		os4.writeObject(avgZScorePattern);
		os4.flush();
		os4.reset();
		os4.close();		
		Path temp5 = new Path(hadoopDirectory + "out_npairsj_ser/", "avgNoisePattern_" + value.toString().trim());
		ObjectOutputStream os5 = new ObjectOutputStream(fs4.create(temp5));
		os5.writeObject(avgNoisePattern);
		os5.flush();
		os5.reset();
		os5.close();
		Path temp6 = new Path(hadoopDirectory + "out_npairsj_ser/", "ppTrueClass_" + value.toString().trim());
		ObjectOutputStream os6 = new ObjectOutputStream(fs4.create(temp6));
		os6.writeObject(ppTrueClass);
		os6.flush();
		os6.reset();
		os6.close();
		Path temp7 = new Path(hadoopDirectory + "out_npairsj_ser/", "sqrdPredError_" + value.toString().trim());
		ObjectOutputStream os7 = new ObjectOutputStream(fs4.create(temp7));
		os7.writeObject(sqrdPredError);
		os7.flush();
		os7.reset();
		os7.close();
		Path temp8 = new Path(hadoopDirectory + "out_npairsj_ser/", "predClass_" + value.toString().trim());
		ObjectOutputStream os8 = new ObjectOutputStream(fs4.create(temp8));
		os8.writeObject(predClass);
		os8.flush();
		os8.reset();
		os8.close();
		Path temp9 = new Path(hadoopDirectory + "out_npairsj_ser/", "correctPred_" + value.toString().trim());
		ObjectOutputStream os9 = new ObjectOutputStream(fs4.create(temp9));
		os9.writeObject(correctPred);
		os9.flush();
		os9.reset();
		os9.close();
		Path temp10 = new Path(hadoopDirectory + "out_npairsj_ser/", "ppAllClasses_" + value.toString().trim());
		ObjectOutputStream os10 = new ObjectOutputStream(fs4.create(temp10));
		os10.writeObject(ppAllClasses);
		os10.flush();
		os10.reset();
		os10.close();
		Path temp11 = new Path(hadoopDirectory + "out_npairsj_ser/", "r2_" + value.toString().trim());
		ObjectOutputStream os11 = new ObjectOutputStream(fs4.create(temp11));
		os11.writeObject(r2);
		os11.flush();
		os11.reset();
		os11.close();
		
		Path temp12 = new Path(hadoopDirectory + "out_npairsj_ser/", "corrCoeffs_" + value.toString().trim());
		ObjectOutputStream os12 = new ObjectOutputStream(fs4.create(temp12));
		os12.writeObject(corrCoeffs);
		os12.flush();
		os12.reset();
		os12.close();
		
		Path temp13 = new Path(hadoopDirectory + "out_npairsj_ser/", "noisePattStdDev_" + value.toString().trim());
		ObjectOutputStream os13 = new ObjectOutputStream(fs4.create(temp13));
		os13.writeObject(noisePattStdDev);
		os13.flush();
		os13.reset();
		os13.close();
		
		double tTime = (System.currentTimeMillis() - sTime) / 1000;
		System.out.println("Serialization takes [" + tTime + " s]");
		
		/*******************************************************************/
        
    }
 }
/*
  
  public static class IntSumReducer 
       extends Reducer<Text,IntWritable,Text,IntWritable> {
    //private IntWritable result = new IntWritable();
    public void reduce(Text key, Iterable<IntWritable> values, 
                       Context context
                       ) throws IOException, InterruptedException {

    	//Matrix avgCVScoresTest = new MatrixImpl(3000, 1000).getMatrix();
    	/*
    	double time = System.currentTimeMillis()/ 1000;
		for (IntWritable val : values) {
		//Alan: output runtime log files to HDFS
			double time2 = System.currentTimeMillis()/ 1000;
			Path pt=new Path("/user/hadooplu/RunTimeLog",Double.toString(time)+"_"+val.toString());
			FileSystem fs = FileSystem.get(context.getConfiguration());
			BufferedWriter br=new BufferedWriter(new OutputStreamWriter(fs.create(pt,true)));
			br.write(key.toString() + "/" + val.toString());
			br.close();		  	
		}
		
		*/
		/*
      for (IntWritable val : values) {
    	
    	//for(int i = 0; i < 3; i++){
    		FileSystem fs3 = FileSystem.get(new Configuration());
            Path path3 = new Path("/user/hadooplu/" + key.toString() + "/" +  val.toString()  );
            InputStream in3 = fs3.open(path3);
            ObjectInputStream objReader3 = new ObjectInputStream(in3);
            
            Matrix current_matrix = new MatrixImpl(3000, 1000).getMatrix();
            
            try {
            	
            	current_matrix = (Matrix)objReader3.readObject();

        		avgCVScoresTest = avgCVScoresTest.plus(current_matrix);
        		
            } catch (ClassNotFoundException e) {
      		// TODO Auto-generated catch block
      		e.printStackTrace();
            }
    	}
    	
        //sum += val.get();
        
        
  		/******************Serialize objects*********************
		FileSystem fs4 = FileSystem.get(context.getConfiguration());
		Path temp1 = new Path("/user/hadooplu/output", key.toString());
		ObjectOutputStream os1 = new ObjectOutputStream(fs4.create(temp1));
		os1.writeObject(avgCVScoresTest);
		os1.flush();
		os1.reset();
		os1.close();
		/*******************************************************************/
      
      //result.set(sum);
      //context.write(key, result); 
   // }
  //}

  public static void main(String[] args) throws Exception {
      
	Configuration conf = new Configuration();
    String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
    
    Job job = new Job(conf, "test");
    job.setJarByClass(Test.class);
    job.setMapperClass(TokenizerMapper.class);
    //job.setCombinerClass(IntSumReducer.class);
    //job.setReducerClass(IntSumReducer.class);
    job.setNumReduceTasks(0);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(IntWritable.class);
    FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
    FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));
    
    double start = System.currentTimeMillis();
    System.out.println("starting to move tmp serialized files to HDFS");
    
    //starting to move tmp serialized files to HDFS
    FileSystem hdfsFileSystem = FileSystem.get(new Configuration());

    Path out = new Path(hadoopDirectory + "out");
    if(hdfsFileSystem.exists(out)){
    	hdfsFileSystem.delete(out, true);
    }
    
    Path in = new Path(hadoopDirectory + "Hadoop_input_files");
    if(hdfsFileSystem.exists(in)){
    	hdfsFileSystem.delete(in, true);
    }
    
    Path local1 = new Path("setupParams.ser");
    Path local2 = new Path("splits.ser");
    Path local3 = new Path("Hadoop_input_files/");
   // Path local4 = new Path("dataLoader.ser");
    Path local5 = new Path("fullDataAnalysis.ser");
    Path hdfs = new Path(hadoopDirectory); 
    hdfsFileSystem.copyFromLocalFile(true, local1, hdfs);
    hdfsFileSystem.copyFromLocalFile(true, local2, hdfs);
    hdfsFileSystem.copyFromLocalFile(false, local3, hdfs);
   // hdfsFileSystem.copyFromLocalFile(false, local4, hdfs);    
    hdfsFileSystem.copyFromLocalFile(false, local5, hdfs);    
    
    double end= (System.currentTimeMillis() - start) / 1000;    
    System.out.println("DONE in " + end + "seconds");
    
    //submit hadoop job
    double sTime = System.currentTimeMillis();
    job.waitForCompletion(true);
    double tTime = (System.currentTimeMillis() - sTime) / 1000;
    System.out.println("hadoop job takes: " + tTime + "seconds...");    
        
    //move files from HDFS to local FS
    Path hdfs_out = new Path(hadoopDirectory + "out_npairsj_ser"); 
    ///Path local6 = new Path("/home/alanlu/NPAIRS/");
    Path local6 = new Path(localDirectory);
    hdfsFileSystem.copyToLocalFile(true, hdfs_out, local6);
  }
}