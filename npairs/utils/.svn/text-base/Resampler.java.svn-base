package npairs.utils;

import pls.shared.MLFuncs;
import java.math.BigInteger;
import java.util.Hashtable;
import java.util.Vector;
import java.util.HashSet;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.Collection;
import java.util.ArrayList;
import npairs.NpairsjException;
import npairs.io.NpairsjIO;

import java.util.Arrays;

/** This class contains tools for generating samples from an input array. 
 *  
 * @author anita
 *
 */
public class Resampler {
	
	private int comboNum = 0;
	private int offset = 0;
	private double rangeForDet = 0.9;
	
	private boolean debug = false;
	
	/**  Array of labels indicating split objects.  
       E.g. if splitObj = {1, 1, 2, 2, 3, 3, 4, 4}, 
       then there are 4 objects.
       Each object (within a group, see groupDef 
       below) is considered an atomic unit, so cannot
       be split up in the resampling process.
       Note that different objects belonging 
       to different groups may have the same object 
       labels in input splitObj arg to Resampler constructor;
       but before saving input splitObj info in this variable,
       split objects from different grps with same label are 
       relabelled so that no two groups have same object labels 
	 */
	private int[] splitObj;                          
	
	/**	 2-element array containing number of split objects in each
     split pair (i.e., {n1, n2}, where n1 = no. of objects to be
     included in first ('training') split half; n2 = no. of objects
     to be incl. in second ('test') split half.
	 */
	private int[] splitPartition; 
	
	/**	 Upper bound on the number of split-pair samples to generate.  
     */
	private int numSplits; 
	
	/** Actual number of split-pair samples to generate;
	 *  numSamples = min(numSplits, max possible splits)
	 */
	private int numSamples;
		
	/**  Total number of objects to be split */
	private int nObj;                  
	
	/**	 Total number of groups */
	private int nGrps;                
	
	/** nGrps-length array indicating number of objects in each group
	 */
	private int[] nObjInGrp;            
	
	/**	 2D array (int[nGrps][]) containing, for each group, 
    the array of unique splitObj labels contained in that group.
	 */
	private int[][] objListByGrp;                     
	
	/**	 contains arrays of indices giving locations of each 
	   object label in input 'splitObj'
	 */
	private Hashtable<Integer, int[]> objLocArrays;  
	
	/**	 contains arrays of indices giving locations of each 
	 * group label in input 'groupDef'
	 */
	private Hashtable<Integer, int[]> grpLocArrays;   
	
    /**	 contains group labels in ascending order
     */
	private int[] uniqGrpLabels;    
	
	/** 2D array BigInteger[2][nGrps] giving number of possible combinations of 
	 *  elements within each group, within each split half
	 */
	private BigInteger[][] numCombosPerGrp;
	
	/** 3D array of combos as HashSets: HashSet[nGrps][2].
	 *  For each group and split half, have array of length
	 *   = number of possible unique split-half pairs for curr
	 *  group.  Hence allCombosPerGrp[i][j] may have repeated
	 *  entries for given split half j, but allCombosPerGrp[i]
	 *  will not contain any duplicate split half (ordered) PAIRS.  
	 *  I.e., given allCombosPerGrp[i][0] = {A} and 
	 *                 allCombosPerGrp[i][1] = {B},
	 *  CAN have allComboPerGrp[j][0] = {A}, j != i, but NOT
	 *  with     allCombosPerGrp[j][1] = {B}.
	 *  Note also that in case of equal split halves,
	 *  allCombosPerGrp WILL contain duplicates of the form
	 *  allCombosPerGrp[j][0] = {B}, allCombosPerGrp[j][1] = {A}.
	 */
	private HashSet[][][] allCombosPerGrp;
	
	 /** 2D int array containing number of elements (split objects) from each
	 *  'group' to be included in each split half when resampling data
	 */
	int[][] grpPartitions;
	
	/**  If true, then data should be 'partitioned'
     into {m, 0} split halves (m <= nObj), sampling
     with replacement.  Otherwise cannot have
     empty split half, and sampling is done without
     replacement.
	 */
	//private boolean bootstrap = false; // NOT IMPLEMENTED YET     
	
	/** If true, then each split half contains the same number
	 *  of objects per group, so combos generated for each split
	 *  half are symmetric; number of total possible unique combos
	 *  is therefore divided in two, to eliminate duplication when
	 *  split halves' roles are switched (i.e. 'training' split
	 *  half becomes becomes 'test' split half 
	 *  and vice versa)
	 */
	private boolean equalSplitHalves = true;
	
	private boolean randSelection = true; 
	
	private boolean bootstrap = false;
	
	
	/*************************************************************************/
	
	/** Constructor - no bootstrap flag (so bootstrap won't be done)
	 * 
	 */
	public Resampler (int[] splitObj,
			          int[] groupDef,
			          int[] splitPartition,
			          int numSplits) {
		this(splitObj, groupDef, splitPartition, numSplits, false);
	}
	
	
	/** Constructor (including bootstrap flag)
	 * 
	 */
	public Resampler(int[] splitObj,
					 int[] groupDef,
					 int[] splitPartition,
					 int numSplits,
					 boolean bootstrap) {
		
//		if (bootstrap) {
//			throw new IllegalArgumentException("Bootstrap not implemented yet...");
//		}
		if (groupDef.length != splitObj.length) {
			throw new IllegalArgumentException("Error - groupDef must contain same "
					+ "number of elements as splitObj");
		}
		if (splitPartition.length != 2) {
			throw new IllegalArgumentException("Error - splitPartition must contain "
					+ "exactly 2 elements.");
		}
		
		// get number of groups
		grpLocArrays = CVA.getLabelIndices(groupDef);
		nGrps = grpLocArrays.size();		
		if (debug) {
			System.out.println("Num groups : " + nGrps);
		}
		
		// get number of objects
		uniqGrpLabels = new int[nGrps];
		int grp = 0;
		for (Enumeration<Integer> e = grpLocArrays.keys(); e.hasMoreElements(); ++grp) {
			uniqGrpLabels[grp] = (Integer)e.nextElement();
		}
	    uniqGrpLabels = MLFuncs.sortAscending(uniqGrpLabels);
 
	    // relabel split objects to ensure that objects from different groups 
	    // have different object labels 
		int[] splitObjRelabelled = new int[splitObj.length];
		HashSet<Integer> uniqObjLabels = new HashSet<Integer>();
		int offset = 1; // for creating new split obj labels
		for (int g = 0; g < nGrps; ++g) {
			int[] currGrpLoc = grpLocArrays.get(uniqGrpLabels[g]);
			int[] currSplitObj = MLFuncs.getItemsAtIndices(splitObj, currGrpLoc);
			int[] uniqCurrSplitObj = MLFuncs.unique(currSplitObj);
			for (int label : uniqCurrSplitObj) {
				int [] labelLoc = MLFuncs.find(splitObj, label);
			    Vector<Integer> currGrpLabelLoc = new Vector<Integer>();
			    for (int loc : labelLoc) {
			    	if (MLFuncs.contains(currGrpLoc, loc)) {
			    		currGrpLabelLoc.add(loc);
			    	}
			    }
			    if (uniqObjLabels.contains(label)) {
					// curr label already used in another grp; relabel curr obj
					// before adding to splitObjRelabelled  
				    int newLabel = MLFuncs.max(splitObj) + offset;
				    ++offset;
				    for (int loc : currGrpLabelLoc) {
				    	splitObjRelabelled[loc] = newLabel;
				    }
				    uniqObjLabels.add(newLabel);
				}
				else {
					for (int loc : currGrpLabelLoc) {
						splitObjRelabelled[loc] = label;
					}
					uniqObjLabels.add(label);
				}
			}
		}
//		if (debug) {
//			System.out.println("Split obj relabelled: ");
//			npairs.io.NpairsjIO.print(splitObjRelabelled);
//		}	
		objLocArrays = CVA.getLabelIndices(splitObjRelabelled);
		nObj = objLocArrays.size();
		if (debug) {
			System.out.println("Num Obj: " + nObj);
		}
		
		if (splitPartition[0] + splitPartition[1] > nObj) {
			throw new IllegalArgumentException("Error - sum of elements partitioned " 
				  + "into split \nhalves cannot exceed total number of objects to be "
				  + "split.");
		}
		if (bootstrap == false) {
			if (splitPartition[0] < nGrps || splitPartition[1] < nGrps) {
				throw new IllegalArgumentException("Error - each split half must "
					+ "contain at least one element from each group.");
			}
		}
//		else { // must have splitPartition = {nObj, 0} for bootstrap
//			System.out.println("Bootstrap not implemented yet...");
//			return;
////			if (splitPartition[0] > nObj || splitPartition[1] != 0) {
////				throw new IllegalArgumentException("Error - splitPartition must equal " +
////						"{m, 0} when bootstrapping, \nwhere 0 < m <= num input samples " +
////						"(i.e. split objects).");
////			}
//		}
		
		this.splitObj = splitObjRelabelled;
		this.splitPartition = splitPartition;
		this.numSplits = numSplits;
		this.bootstrap = bootstrap;
	}
	
	
	/** Generates split half samples.
	 * 
	 * @param boolean - if true, then splits are generated
	 *                  uniformly randomly (without replacement);
	 *                  otherwise splits are generated deterministically
	 * @return - 3D array int[2][numSamples][] containing sorted sample
	 *           pairs.  int[0][][] corresp. to first split halves,
	 *           and int[1][][] corresp. to second split halves.
	 *           The elements in the array are indices into original
	 *           splitObj array provided as input arg to Resampler 
	 *           constructor. Each sample split half is sorted into
	 *           ascending order.
	 * 
	 * @throws NpairsjException
	 */
	public int[][][] generateSplits(boolean random) throws NpairsjException {

		// get number of samples from each group to be included in each split half
		grpPartitions = partitionGroups();
		// NOTE if split halves are equal, maxNumCombos will be twice as large
	    // as the true max num unique combos, since getMaxNumCombos() does not account
		// for duplicate split half PAIRS, e.g. if {A, B} is a split half pair,
		// then {B, A} will not be excluded in maxNumCombos
		BigInteger maxNumCombos = getMaxNumCombos(); 
		BigInteger bigNumSamples;
		if (equalSplitHalves) {
			// actually only have maxNumCombos/2 possible unique split half pairs
			bigNumSamples = (maxNumCombos.divide(new BigInteger("2"))).min(
					new BigInteger(Integer.toString(numSplits)));
		}
		else {
			bigNumSamples = maxNumCombos.min(new BigInteger(Integer.toString(numSplits)));
		}
		// note that bigNumSamples will necessarily be in int range now (since numSplits
		// is in int range)
		numSamples = bigNumSamples.intValue();
		
//		if (debug && !bootstrap) {
//			System.out.println("No. samples: " + numSamples);
//		}
				
		int[][][] samples;
			
		// check that maxNumCombos is within int range
		int maxInt = new Integer(Integer.MAX_VALUE);
		int intmaxNumCombos = maxInt;
		
		if (!(maxNumCombos.compareTo(new BigInteger(Integer.toString(maxInt))) > 0)) {
			intmaxNumCombos = maxNumCombos.intValue();
		}
		
//		if (debug) {
//			System.out.println("maxNumCombos: " + intmaxNumCombos);
//			System.out.println("maxNumCombos = maxInt: " + (intmaxNumCombos == maxInt));
//		}
		
		if (bootstrap) {
			// can do as many bootstrap samples as desired
//			if (debug) {
//				System.out.println("No. samples: " + numSplits);
//			}
			samples = generateRandom(numSplits);
		}
		
		else if ((random) && (numSamples <= rangeForDet * intmaxNumCombos)) {
			
			// can only do as many samples as there are possible unique samples
			samples = generateRandom(numSamples);
			
//			if (debug){
//				System.out.println("Used random method to generate samples...");
//				randSelection = true;
//			}
		}
		
		else {
			if (debug) {
				System.out.println("Using deterministic method to generate samples...");
				randSelection = false;
			}
			
			samples = generateDeterministic(numSamples, intmaxNumCombos);
			randSelection = false;
		}
		          
		if (debug) {
			System.out.println("Finished generating samples...");
			System.out.println("Combos Split 1: ");
			npairs.io.NpairsjIO.print(samples[0]);
			System.out.println("Combos Split 2: ");
			npairs.io.NpairsjIO.print(samples[1]);
		}			
		
		return samples;
	}
	
/*****************************************************************************
 * 
 * 	Helper methods
 * 
 *****************************************************************************/
	
	/** Returns 2D int array containing number of elements (split objects) from 
	 *  each 'group' to be included in each split half when resampling data.
	 *  (Technique somewhat follows Jon Anderson's npairs_splits.pro IDL code.)
	 *  
	 * @return 2D int array - 2 rows X numGroups cols. 
	 *                      - 1st row == 1st split half; 2nd row == 2nd split half
	 *                      - groups are in ascending label order   
	 * @throws NpairsjException if partitioning group elements is unsuccessful 
	 *         after 10000 tries
	 */
	private int[][] partitionGroups() throws NpairsjException {		
		// number of obj in each group:
		nObjInGrp = new int[nGrps];
		objListByGrp = new int[nGrps][];
		for (int g = 0; g < nGrps; ++g) {
			int[] currGrpLocs = grpLocArrays.get(uniqGrpLabels[g]);		
			int[] currGrpObjArray = MLFuncs.getItemsAtIndices(splitObj, currGrpLocs);
			objListByGrp[g] =  MLFuncs.unique(currGrpObjArray);
			nObjInGrp[g] = objListByGrp[g].length;
		}
		if (debug) {
			System.out.println("Num objects in each group: ");
			npairs.io.NpairsjIO.print(nObjInGrp);
		}
		
		// NOTE - proportion of objects from each grp in each sample is the same as 
		// proportion of objects from each grp in total no. of objects.
		// Determine proportions:
		double[][] nGrpObjPerSplit = new double[2][nGrps]; // 1st row == no in 1st 
		                                                // split half
                                                        // 2nd row == no in 2nd 
		                                                // split half
		for (int g = 0; g < nGrps; ++g) {
			double grpFactor = (double)nObjInGrp[g]/nObj;
			nGrpObjPerSplit[0][g] = grpFactor * splitPartition[0]; 
			nGrpObjPerSplit[1][g] = grpFactor * splitPartition[1];
		}	

		if (debug) {
			System.out.println("Num obj in each split half: {" + splitPartition[0] + 
					", " + splitPartition[1] + "}");
			System.out.println("Num obj from each group in each split half " +
					"(Col = grp; row = split half):");
			npairs.io.NpairsjIO.print(nGrpObjPerSplit);
		}
		
		// check that correct total no. of objects is assigned to each split half
		int[] flag = new int[nGrps];
		int[][] subPartitions = new int[2][nGrps];
		for (int i = 0; i < 2; ++i) { // for each split half
			int nTry = 0;
			int[] subPartn = new int[nGrps];
			for (int g = 0; g < nGrps; ++g) {
				subPartn[g] = (int)Math.round(nGrpObjPerSplit[i][g]);
			}
			if (debug) {
				System.out.println("Partitions for split half " + (i+1) + ": ");
				NpairsjIO.print(subPartn);
			}
			int sumObjInCurrSplitHalf = MLFuncs.sum(subPartn); 
			boolean numObjectsMatch = (sumObjInCurrSplitHalf == splitPartition[i]);
			
			int[] tmp = new int[nGrps];
			while (!numObjectsMatch) {
				if (debug) {
					System.out.println("Num els in split half " + (i+1) + ": " + sumObjInCurrSplitHalf);
					System.out.println("Incorrect number of els so doing reassignment...");
				}
				System.arraycopy(subPartn, 0, tmp, 0, nGrps);
				if (sumObjInCurrSplitHalf < splitPartition[i]) {
					// add 1 to all elements in curr split half:
					for (int g = 0; g < nGrps;  ++g) {
						tmp[g] += 1;
					}
				}
				else {
					for (int g = 0; g < nGrps; ++g) {
							tmp[g] -= 1;
					}
				}
				// don't use grps that have already been adjusted:
				for (int g = 0; g < nGrps; ++g) {
					if (flag[g] == 1) {
						tmp[g] = 999999;
					}
				}
				// find grp(s) that would be least affected by adding/subtracting 1
				double[] diff = new double[nGrps];
				for (int g = 0; g < nGrps; ++g) {
					// NOTE comparison is with orig calculated double nGrpObjPerSplit
					// value instead of rounded int version 
					// (see IDL npairs_split.pro code)
					diff[g] = Math.abs(tmp[g] - nGrpObjPerSplit[i][g]); 
				}
				double min = MLFuncs.min(diff);
				int[] whereMin = MLFuncs.find(diff, min);
				// adjust a random whereMin grp
				int loc = (int)Math.floor(Math.random() * whereMin.length);  // 0 <= Math.random < 1 so OK
				if (debug) {
					System.out.println("Adjusting grp " + (whereMin[loc] + 1) + "...");
				}
				subPartn[whereMin[loc]] = tmp[whereMin[loc]];
				if (debug) {
					System.out.println("Don't adjust group no. " + (whereMin[loc] + 1) + " again...");
				}
				flag[whereMin[loc]] = 1; // don't adjust this grp again 
				
				// give up after 10000 tries
				nTry += 1;
				if (nTry == 10000) {
					throw new NpairsjException("Error - could not determine partition "
							+ "\nof group elements when resampling");
				}
				sumObjInCurrSplitHalf = MLFuncs.sum(subPartn); 
				numObjectsMatch = (sumObjInCurrSplitHalf == splitPartition[i]);
		    }
			subPartitions[i] = subPartn;	
			
		}
		if (debug) {
			System.out.println("Adjusted partition: ");
			NpairsjIO.print(subPartitions);
		}
		
		// make sure each group's subpartition has no more than the number of elements that should
		// be used from that group
		double[] nObjInGrpPartns = new double[nGrps];
		for (int g = 0; g < nGrps; ++g) {
			nObjInGrpPartns[g] = nGrpObjPerSplit[0][g] + nGrpObjPerSplit[1][g];
		}
		if (debug) {
			System.out.println("Num obj in group partitions: ");
			NpairsjIO.print(nObjInGrpPartns);
		}
        subPartitions = adjustPartitions(subPartitions, nObjInGrpPartns);
		
		if (debug) {
			System.out.println("Num obj from each grp after reassignment: ");
			npairs.io.NpairsjIO.print(subPartitions);
		}
		
		for (int i = 0; i < 2; ++i) {
			if ((splitPartition[i] != 0) && (MLFuncs.contains(subPartitions[i], 0))) {
				throw new NpairsjException("Error - must have at least one object "
						+ "from each \ngroup in each split half.  Include more data!");
			}
		}
		return subPartitions;		
	}
	
	/** Adjusts partition sizes if input subPartitions contain too
	 *  many elements for any groups.
	 *
	 * @param subPartitions - no. of elements from each group in
	 *                        each split half
	 *                      - array size: [2][nGrps] where 1st row
	 *                        corresponds to 1st split half and 2nd row corresponds
	 *                        to 2nd split half.
	 * @param maxNObjInGrpPartn total number of elements from each group to be incl. in
	 *                       a partition (as double; actually fraction of total no. els to
	 *                       be included, e.g. if 5 of 19 total input els belong to grp g, and
	 *                       the partition is input as (4,3), then 
	 *                       maxNObjInGrpPartn[g] = (5/19)*4 + (5/19)*3)
	 * @throws NpairsjException if partitioning adjustment is unsuccessful because no
	 *                          grp exists with fewer than ceiling(maxNObjInGrpPartn) els
	 * @return Adjusted partitions. No. of elements will now be correct for
	 *         each split half and group.
	 */
	private int[][] adjustPartitions(int[][] subPartitions, double[] maxNObjInGrpPartn) throws NpairsjException {
		if (debug) {
			System.out.println("Adjusting partitions...");
		}
		double[] partSzDiffDbl = new double[nGrps]; 
		int[] partSzDiffInt = new int[nGrps];
		boolean discrepancy = false;
		for (int g = 0; g < nGrps; ++g) {
			int partSz = subPartitions[0][g] + subPartitions[1][g];
			partSzDiffInt[g] = partSz - (int)Math.ceil(maxNObjInGrpPartn[g]); // comparison is with upper bound
																			  // of eligible no. els in this grp
																			  // partition
			partSzDiffDbl[g] = partSz - maxNObjInGrpPartn[g];  

			if (partSzDiffInt[g] > 0) {
				// too many els in partition
				discrepancy = true;
			}

		}

		if (discrepancy) {
			if (debug) {
				System.out.println("Found discrepancy...");
			}
			// there must be a partition with fewer than max possible els
			// for each partition with
			// too many
			int[] fewerThanMax = new int[nGrps];
			int[] tooMany = new int[nGrps];
			for (int g = 0; g < nGrps; ++g) {
				if (partSzDiffInt[g] > 0) {
					tooMany[g] = 1;
				}
				else if (partSzDiffInt[g] < 0) {
					fewerThanMax[g] = 1;
				}
			}

			if (MLFuncs.sum(tooMany) > 0) {

				if (MLFuncs.sum(fewerThanMax) > 0) {
					// find grp best suited to having el added
					int minGrp = MLFuncs.find(partSzDiffDbl, MLFuncs.min(partSzDiffDbl))[0];

					// take away one el from a grp with too many and add it to
					// minGrp
					for (int g = 0; g < nGrps; ++g) {
						if (tooMany[g] == 1) {
							if (debug) {
								System.out.println("too many els from grp " + (g+1) + "!");
							}
							// take el from split half with the greater no.
							// of els from this grp if possible
							boolean adjustSplitHalf1 = true;
							if (subPartitions[1][g] > subPartitions[0][g]) {
								--subPartitions[1][g];
								adjustSplitHalf1 = false;
							}
							else if (subPartitions[1][g] < subPartitions[0][g]) {
								--subPartitions[0][g];
							}
							else {
								// check which split half, if any, has fewer els in 
								// minGrp

								if (subPartitions[0][minGrp] < subPartitions[1][minGrp]) {
									// first split half has fewer els so adjust this
									// split half (in too-big group g)
									// and add el to minGrp
									--subPartitions[0][g];
									++subPartitions[0][minGrp];
								}
								else {
									--subPartitions[1][g];
									++subPartitions[1][minGrp];				
								}
								break;
							}
						}
					}

				}
				else {
					throw new NpairsjException("Error - could not adjust group partitions when resampling.");
				}

			}
			subPartitions = adjustPartitions(subPartitions, maxNObjInGrpPartn);	
		}



		return subPartitions;
	}
	
	
	/** Generates 'numSamples' samples of split half pairs by generating all 'maxNumCombos'
	 *  split half pairs systematically and then randomly choosing 'numSamples' of them.
	 * 
	 */
	 private int[][][] generateDeterministic(int numSamples, int maxNumCombos) {
		 
		 // first generate all combos for each group, for each split half:
		 HashSet[][] grpCombosSplit1 = new HashSet[nGrps][];
		 // second split half will contain set of possible combos GIVEN EACH
		 // INSTANCE OF SPLIT-1 SAMPLE FROM EACH GROUP. Second dim will be size
		 // of num combos in split half 1 corresp. to given group; third dim
		 // will be num combos possible in split half 2 from curr. group
		 HashSet[][][] grpCombosSplit2 = new HashSet[nGrps][][];
		for (int g = 0; g < nGrps; ++g) {
			 // since maxNumCombos is an int value, so are numCombosPerGrp vals
			 grpCombosSplit1[g] = new HashSet[numCombosPerGrp[0][g].intValue()];
			 grpCombosSplit2[g] = 
				 new HashSet[numCombosPerGrp[0][g].intValue()][(numCombosPerGrp[1][g]).intValue()];
	 
			 // get split 1 combos for curr grp
			 CombinationGenerator cg1 = new CombinationGenerator(nObjInGrp[g], 
					 grpPartitions[0][g], numCombosPerGrp[0][g].intValue()); 
			 grpCombosSplit1[g] = cg1.generateCombos(objListByGrp[g]);
			 
			 // get split 2 combos for curr grp
			 int nObjLeftInGrp = nObjInGrp[g] - grpPartitions[0][g];
			 for (int c = 0; c < grpCombosSplit1[g].length; ++c) {
				 HashSet<?> combo = grpCombosSplit1[g][c];
				 CombinationGenerator cg2 = new CombinationGenerator(nObjLeftInGrp, 
						 grpPartitions[1][g], numCombosPerGrp[1][g].intValue());
				 int[] objLeftInGrp = new int[nObjLeftInGrp];
				 int j = 0;
				 for (int i : objListByGrp[g]) {
					 if (!combo.contains(i)) {
						 objLeftInGrp[j] = i;
						 ++j;
					 }
				 }
				 grpCombosSplit2[g][c] =  cg2.generateCombos(objLeftInGrp);
			 }

		}
		// generate all split-half pairs of combos for each group
		allCombosPerGrp = new HashSet[nGrps][2][];
		for (int g = 0; g < nGrps; ++g) {
			int nCombo1 = numCombosPerGrp[0][g].intValue();
			int nCombo2 = numCombosPerGrp[1][g].intValue();
			allCombosPerGrp[g][0] = new HashSet[nCombo1 * nCombo2];
			allCombosPerGrp[g][1] = new HashSet[nCombo1 * nCombo2];
			for (int c1 = 0; c1 < nCombo1; ++c1) {
				HashSet<?> currGrpComboSplit1 = grpCombosSplit1[g][c1];
				for (int c2 = 0; c2 < nCombo2; ++c2) {
					HashSet<?> currGrpComboSplit2 = grpCombosSplit2[g][c1][c2];
					allCombosPerGrp[g][0][c2 + (c1 * nCombo2)] = currGrpComboSplit1;
					allCombosPerGrp[g][1][c2 + (c1 * nCombo2)] = currGrpComboSplit2;				
				}
			}
		}
//		if (debug) {
//			System.out.println("All combos per grp:");
//			for (int g = 0; g < nGrps; ++g) {
//				System.out.println("Grp " + g + ":");
//				npairs.io.NpairsjIO.print(allCombosPerGrp[g]);
//			}
//		}
		
		// combine combos across groups
		int[][][] allCombos = combineGrpCombos(maxNumCombos);
		
		//just return first numSamples combos
		int[][][] tmpSelectedCombos = new int[2][numSamples][];
		for (int i = 0; i < 2; ++i) {
			for (int s = 0; s < numSamples; ++s) {
				tmpSelectedCombos[i][s] = allCombos[i][s];
			}
		}
		
		// record locations of chosen objects in splitObj array
		int[][][] selectedCombos = new int[2][numSamples][];
		
		for (int i = 0; i < 2; ++i) {
			
			for (int s = 0; s < numSamples; ++s) {
				
				Vector<Integer> splitObjLocs = new Vector<Integer>();
				for (int label : tmpSelectedCombos[i][s]) {
					
					int[] currSplitObjLocs = MLFuncs.find(splitObj, label);
//					if (debug) {
//						System.out.println("For label " + label + ", found " + currSplitObjLocs.length +
//							" locations: ");
//						npairs.io.NpairsjIO.print(currSplitObjLocs);
//					}
					for (int loc : currSplitObjLocs) {						
							splitObjLocs.add(loc);												
					}
				}
			
//				if (debug) {
//					System.out.println("Split Obj Locs - split half " + i + " sample no. " + s + ":");
//					System.out.println(splitObjLocs.toString());
//				}
				selectedCombos[i][s] = new int[splitObjLocs.size()]; 
				for (int j = 0; j < splitObjLocs.size(); ++j) {
					selectedCombos[i][s][j] = splitObjLocs.get(j);
					
				}
				// sort split obj indices for curr combo
				selectedCombos[i][s] = MLFuncs.sortAscending(selectedCombos[i][s]);
			}
		}
		
		
//		if (debug) {
//			System.out.println("Returning " + numSamples + " samples!");
//		}
		return selectedCombos;
		
	}
	 
	 
	 private int[][][] combineGrpCombos(int maxNumCombos) {
		 int[][][] tmpAllCombos = new int[2][maxNumCombos][];
		 
		 HashSet[] currCombos = new HashSet[2];
		 for (int i = 0; i < 2; ++i) {
			 currCombos[i] = new HashSet<Integer>();
		 }
		 
		 
		 tmpAllCombos = addCombos(tmpAllCombos, nGrps, offset, currCombos, maxNumCombos);
		 
		 int[][][] allCombosFinal;
		 if (equalSplitHalves) {
			 // eliminate duplicate split-half pairs
			 int nUniqCombos = maxNumCombos / 2;
			 allCombosFinal = new int[2][nUniqCombos][];
			 
			 int count = 0;
			 for (int c = 0; c < maxNumCombos; ++c) {
				 boolean isDup = false;
				 int[][] currSplitHalfPair = new int[2][];
				 currSplitHalfPair[0] = tmpAllCombos[0][c];
				 currSplitHalfPair[1] = tmpAllCombos[1][c];
				 // check whether curr pair has already been 
				 // added to final combos
				 for (int s = 0; s < c; ++s) {
					 if (Arrays.equals(tmpAllCombos[0][s], currSplitHalfPair[0])
							 &&
						 Arrays.equals(tmpAllCombos[1][s], currSplitHalfPair[1])) {
						 isDup = true;						 
					 }
					 else if (Arrays.equals(tmpAllCombos[1][s], currSplitHalfPair[0])
							 &&
							  Arrays.equals(tmpAllCombos[0][s], currSplitHalfPair[1])) {
						 isDup = true;
					 }
				 }
				 
				 if (!isDup) {
					 for (int i = 0; i < 2; ++i) {
						 allCombosFinal[i][count] = currSplitHalfPair[i];
					 }
					 ++count;
				 }
			 }
		 }
		 else {
			 allCombosFinal = tmpAllCombos;
		 }
		 return allCombosFinal;
	}
	 
	 
	 
	 /** Recursive method to create all combos - used by combineGrpCombos(...)
	  * REQUIRED: currCombos is 2D HashSet<Integer> array, with both HashSets 
	  * initialized
	  */
	 private int[][][] addCombos(int[][][] allCombos, int numGrps, int offset, 
			 HashSet[] currCombos, int maxNumCombos) {
//		 if (debug) {
//			 System.out.println("NUMGRPS: " + numGrps);
//		 }
		 int nCurrGrpCombos = allCombosPerGrp[numGrps - 1][0].length;
//		 if (debug) {
//			 System.out.println("Num Combos Grp " + numGrps + ": " + nCurrGrpCombos);
//		 }
		 
		 if (numGrps > 1) {
			 
			 int nPrevGrpCombos = allCombosPerGrp[numGrps - 2][0].length;
			 int ng = numGrps - 1;
			 while (ng > 1) {
				 nPrevGrpCombos *= allCombosPerGrp[ng - 2][0].length;
				 --ng;
			 }
			 
			 for (int j = 0; j < nCurrGrpCombos; ++j) {
				 offset += j * nPrevGrpCombos;
//				 if (debug) {
//					 System.out.println("Offset (group no. " + numGrps + ", combo no. " + j + "): ");
//					 System.out.println(offset);
//				 }
				 
				 for (int i = 0; i < 2; ++i) {
					 currCombos[i].addAll(allCombosPerGrp[numGrps - 1][i][j]);			
				 }
//				 if (debug) {
//					 System.out.println("Calling allCombos for grp no. " + numGrps + "... ");
//				 }
				 allCombos = addCombos(allCombos, numGrps - 1, offset, currCombos, maxNumCombos);
				 offset -= j * nPrevGrpCombos;
				 for (int i = 0; i < 2; ++i) {
					 // can 'remove all' because all other elements in currCombos[i]
					 // are from other groups, and therefore disjoint from curr group's
					 // elements
					 currCombos[i].removeAll(allCombosPerGrp[numGrps - 1][i][j]);				 
				 }
			 }
		 }
		 
		 if (numGrps == 1) {
			 for (int j = 0; j < nCurrGrpCombos; ++j) {
				 for (int i = 0; i < 2; ++i) {
					 currCombos[i].addAll(allCombosPerGrp[numGrps - 1][i][j]);				 
					 int[] currSplitCombo = new int[currCombos[i].size()];
					 Iterator<?> iter = currCombos[i].iterator();
					 int count = 0;
					 while (iter.hasNext()) {
						 currSplitCombo[count] = (Integer)iter.next();;
						 ++count;
					 }
					 
//					 if (debug) {
//						 try {
//							 String outFile = "/home/anita/workspace/PLSwithNPAIRS/localTestData/allCombos.txt";
//						     java.io.BufferedWriter out = 
//					 			new java.io.BufferedWriter(new java.io.FileWriter(outFile, true));
//						     String log = "" + comboNum + ": Setting allCombos split half " 
//							 + i + ", combo no. " + (offset + j) + "...";
//						     out.write(log + "\n");
//						     out.close();
//						     
//						 } 
//						 catch (java.io.IOException e) {
//							 e.printStackTrace();					 
//						 }
//						 						 
//					 }	
					 if (i%2 != 0) ++comboNum;
					 allCombos[i][offset + j] = currSplitCombo;
					 currCombos[i].removeAll(allCombosPerGrp[numGrps - 1][i][j]); 
				 }
			 }	 
		 }
		 
		 return allCombos;
	 }
	 
	 
	/**Generates 'numSamples' uniformly random samples of split half pairs.  
	 * If generated random sample is a duplicate, then it is discarded and a 
	 * new split half pair is generated.  A duplicate of a given split half
     * pair {A1, A2} could be  
	 * either of the form {A1, A2} or {A2, A1}.  
	 * 
	 * @param numSamples - no. of sample split half pairs to generate
	 * @return - 3D array int[2][numSamples][] containing sorted sample
	 *           pairs.  int[0][][] corresp. to first split halves,
	 *           and int[1][][] corresp. to second split halves.
	 *           The elements in the array are indices into original
	 *           splitObj array provided as input arg to Resampler 
	 *           constructor. Each sample split half is sorted into
	 *           ascending order.
	 */
	private int[][][] generateRandom(int numSamples) {
//		HashSet[] sampleSets1 = new HashSet[numSamples];
//		HashSet[] sampleSets2 = new HashSet[numSamples];
		ArrayList<Integer>[] sampleSets1 = new ArrayList[numSamples];
		ArrayList<Integer>[] sampleSets2 = new ArrayList[numSamples];
		int count = 0;
		while (count < numSamples) {
//			HashSet[] currSample = new HashSet[2];
			ArrayList<Integer>[] currSample = new ArrayList[2];
//			currSample[0] = new HashSet<Integer>(splitPartition[0]);
//			currSample[1] = new HashSet<Integer>(splitPartition[1]);
			currSample[0] = new ArrayList<Integer>(splitPartition[0]);
			currSample[1] = new ArrayList<Integer>(splitPartition[1]);
			for (int g = 0; g < nGrps; ++g) {
				int[] currGrpObjects = objListByGrp[g]; 
//				if (debug) {
//				System.out.println("Obj list for grp " + g + ":");
//				npairs.io.NpairsjIO.print(currGrpObjects);
//				}
//				HashSet<Integer> objInds1 = 
//					new HashSet<Integer>(grpPartitions[0][g]);
//				HashSet<Integer> objInds2 = 
//					new HashSet<Integer>(grpPartitions[1][g]);
				Collection<Integer> objInds1, objInds2;
				if (bootstrap) {
					// elements will be duplicated
					objInds1 = new ArrayList<Integer>(grpPartitions[0][g]);
					objInds2 = new ArrayList<Integer>(grpPartitions[0][g]);
				}
				else {
					objInds1 = new HashSet<Integer>(grpPartitions[0][g]);
					objInds2 = new HashSet<Integer>(grpPartitions[1][g]);
				}
				
				Vector<Integer> splitObjLocs1 = new Vector<Integer>();
				Vector<Integer> splitObjLocs2 = new Vector<Integer>();
				// randomly generate indices into currGrpObjects for objInds1
				while(objInds1.size() < grpPartitions[0][g]) {
					int nextLoc = (int)Math.floor(nObjInGrp[g] * Math.random());
//					if (debug) {
//					System.out.println("Nextloc split1: " + nextLoc);
//					}
					nextLoc = Math.min(nextLoc, nObjInGrp[g] - 1);
					objInds1.add(nextLoc);
				}
				// record locations of chosen objects in splitObj array
				for (int loc : objInds1) {
					int[] splitObjLoc = MLFuncs.find(splitObj, currGrpObjects[loc]);
					for (int i : splitObjLoc) {
						splitObjLocs1.add(i);
					}
				}				

//				if (debug) {
//					System.out.println("Locs1 - group " + g + ": ");
//					for (Integer i : splitObjLocs1) System.out.print(i + " ");
//					System.out.println();
//				}

				if (!bootstrap) {
					// randomly generate indices for objInds2, excluding objInds1 
					// indices
					while (objInds2.size() < grpPartitions[1][g]) {
						int nextLoc = (int) Math.floor(nObjInGrp[g] * Math.random());
						//					if (debug) {
						//					System.out.println("Nextloc split2: " + nextLoc);
						//					}
						nextLoc = Math.min(nextLoc, nObjInGrp[g] - 1);
						if (!objInds1.contains(nextLoc)) {
							objInds2.add(nextLoc);
						}
					}
				}
				else {
					// in bootstrap, all elements not in the first (training)
					// partition belong in the 2nd (test) partition
					for (int i = 0; i < nObjInGrp[g]; ++i) {
						if (!objInds1.contains(i)) {
							objInds2.add(i);		
						}
					}	
				}
				
//				if (debug) {
//					System.out.println("Obj inds 1: " + objInds1.toString());
//					System.out.println("Obj inds 2: " + objInds2.toString());
//				}
//				
				// find locations of chosen obj in splitObj array
				for (int loc : objInds2) {
					int[] splitObjLoc = MLFuncs.find(splitObj, currGrpObjects[loc]);
					for (int i : splitObjLoc) {
						splitObjLocs2.add(i);
					}
				}
//				if (debug) {
//					System.out.println("Locs2 - group " + g + ": ");
//					for (Integer i : splitObjLocs2) System.out.print(i + " ");
//					System.out.println();
//				}

				// record selections for current group and sample no.
				for (Integer i : splitObjLocs1) {
					currSample[0].add(i);
				}
				for (Integer j : splitObjLocs2) {
					currSample[1].add(j);
				}
				
//				if (debug) {
//					System.out.println("1st split half currSample - grp " + g + ":");
//					System.out.println(currSample[0].toString());
//					System.out.println("2nd split half currSample - grp " + g + ":");
//					System.out.println(currSample[1].toString());
//				}
			}
			
			if (debug) {
				System.out.println("[Before sorting] curr samp 1st split: " + currSample[0]);
				System.out.println("[Before sorting] curr samp 2nd split: " + currSample[1]);
			}
			currSample = sortCurrSample(currSample);
			
			boolean isDup = false;
			if (!bootstrap) {
				// if curr sample is duplicate, discard and try again
				for (int i = 0; i < count; ++i) {
					if (debug) {
						System.out.println("Comparing samples:");
						System.out.println("Curr samp 1st split: " + currSample[0]);
						System.out.println("Sample set 1st split: " + sampleSets1[i]);
						
						System.out.println("Curr samp 2nd split: " + currSample[1]);
						System.out.println("Sample set 2nd split: " + sampleSets2[i]);
					}
					if (currSample[0].equals(sampleSets1[i])) {
						if (currSample[1].equals(sampleSets2[i])) {
							isDup = true;
							if (debug) {
								System.out.println("DUPLICATE!!! - sample no. " + count);
								System.out.println("Curr samp 1: " + currSample[0]);
								System.out.println("Curr samp 2: " + currSample[1]);
								System.out.println("Orig samp 1: " + sampleSets1[i]);
								System.out.println("Orig samp 2: " + sampleSets2[i]);
							}
						}
					}
					if (currSample[0].equals(sampleSets2[i])) {
						if (currSample[1].equals(sampleSets1[i])) {
							isDup = true;
							if (debug) {
								System.out.println("DUPLICATE!!! - sample no. " + count);
								System.out.println("Curr samp 1: " + currSample[0]);
								System.out.println("Curr samp 2: " + currSample[1]);
								System.out.println("Orig samp 1: " + sampleSets1[i]);
								System.out.println("Orig samp 2: " + sampleSets2[i]);
							}
						}
					}
				}
			}
			
			if (!isDup) {				
				sampleSets1[count] = currSample[0];
				sampleSets2[count] = currSample[1];
				count++;
			}
		}
		
		
		// Don't need to sort samples (already sorted); just plug them
		// into 3D array 'sortedSamples'
		int[][][] sortedSamples = new int[2][numSamples][];
		for (int i = 0; i < numSamples; ++i) {
			Integer[] tmpSamp1 = new Integer[sampleSets1[i].size()];
			Integer[] tmpSamp2 = new Integer[sampleSets2[i].size()];
			tmpSamp1 = (Integer[])sampleSets1[i].toArray(tmpSamp1);
			tmpSamp2 = (Integer[])sampleSets2[i].toArray(tmpSamp2);
			sortedSamples[0][i] = new int[tmpSamp1.length];
			sortedSamples[1][i] = new int[tmpSamp2.length];
			for (int j = 0; j < tmpSamp1.length; ++j) {
				sortedSamples[0][i][j] = tmpSamp1[j];
			}
			for (int j = 0; j < tmpSamp2.length; ++j) {
				sortedSamples[1][i][j] = tmpSamp2[j];
			}
//			sortedSamples[0][i] = MLFuncs.sortAscending(sortedSamples[0][i]);
//			sortedSamples[1][i] = MLFuncs.sortAscending(sortedSamples[1][i]);
		}

		if (debug) {
//			System.out.println("First split half samples: ");
//			for (int i = 0; i < numSamples; ++i) {
//				npairs.io.NpairsjIO.print(sortedSamples[0][i]);
//				System.out.println();
//			}
//			System.out.println("Second split half samples: ");
//			for (int i = 0; i < numSamples; ++i) {
//				npairs.io.NpairsjIO.print(sortedSamples[1][i]);
//				System.out.println();
//			}
		}
		return sortedSamples;
	}
	
	
	private ArrayList<Integer>[] sortCurrSample(ArrayList<Integer>[] currSample) {
		int sampSz1 = currSample[0].size();
		int sampSz2 = currSample[1].size();
		Integer[] currSamp1 = new Integer[sampSz1];
		Integer[] currSamp2 = new Integer[sampSz2];
		currSamp1 = currSample[0].toArray(currSamp1);
		currSamp2 = currSample[1].toArray(currSamp2);
		int[] intCurrSamp1 = MLFuncs.sortAscending(currSamp1);
		int[] intCurrSamp2 = MLFuncs.sortAscending(currSamp2);
		currSample[0] = new ArrayList<Integer>(sampSz1);
		for (int i1 = 0; i1 < sampSz1; ++i1) {
			currSample[0].add(intCurrSamp1[i1]);
		}
		currSample[1] = new ArrayList<Integer>(sampSz2);
		for (int i2 = 0; i2 < sampSz2; ++i2) {
			currSample[1].add(intCurrSamp2[i2]);
		}
	
		return currSample;
	}


	/** Calculates n choose r (binomial coefficient).
	 *  Uses algo. described by Bruno Preiss, P.Eng
	 * (see http://www.brpreiss.com/books/opus5/html/page460.html).
	 * 
	 * @param int n - size of group of elements to be sampled from
	 * @param int r - no. of elements in each sample
	 * @return n choose r in BigInteger format
	 */
	private static BigInteger binom(int n, int r) {
		BigInteger[] b = new BigInteger[n + 1];
		b[0] = BigInteger.ONE;
		for (int i = 0; i <= n; ++i) {
			b[i] = BigInteger.ONE;
			for (int j = i - 1; j > 0; --j) {
				b[j] = b[j].add(b[j - 1]);
			}
		}
		return b[r];
	}
		
	/** Returns max possible number of combinations, given input group partition info
	 * 
	 * @return max possible no. of combos, as 
	 *         BigInteger, since int is likely to be out of range 
	 */
	private BigInteger getMaxNumCombos() {
		// num combos possible for each split half: 
		BigInteger[] numCombos = {BigInteger.ONE, BigInteger.ONE}; 
		// num combos possible for each group within each split half:
		numCombosPerGrp = new BigInteger[2][nGrps]; 
		
		for (int g = 0; g < nGrps; ++g) {
			int grpSz = nObjInGrp[g];
			int nObjUsed = 0;
			for (int i = 0; i < 2; ++i) { 
				int currNObj = grpPartitions[i][g];
				grpSz = grpSz - nObjUsed;
				nObjUsed = currNObj; 
				// compute number of combinations of 'grpSz' things 'currNObj' at a time
				BigInteger nCombos = binom(grpSz, currNObj);
				numCombosPerGrp[i][g] = nCombos;
//				if (debug) {
//					System.out.println("C(" + grpSz + ", " + currNObj + ") =  " 
//							+ nCombos);
//				}
				numCombos[i] = numCombos[i].multiply(nCombos);
			}
		}	
//		if (debug) {
//			System.out.println("Num combos possible per split half: ");
//			System.out.print(numCombos[0] + "\n" + numCombos[1]);
//			System.out.println();
//		}
		BigInteger maxPossCombos = numCombos[0].multiply(numCombos[1]);
		for (int i = 0; i < nGrps; ++i) {
			if (grpPartitions[0][i] != grpPartitions[1][i]) {
				equalSplitHalves = false;
			}
		}

//		if (debug) {
//			System.out.println("Max possible splits (getMaxNumCombos()): " + maxPossCombos);
//			System.out.println("User-defined num splits: " + numSplits);
//		}
		return maxPossCombos;
	}
	
	/** Returns number of split-half pairs generated by this Resampler
	 *  ( = min(numSplits, max possible num splits)
	 * @return
	 */
	public int getNumSamples() {
	    return numSamples;
	}
	    
	public boolean getRand(){
	        return randSelection;
	}
		

	// For testing
	
	
	public static void main(String[] args) {
		// 1. test 10,10,10,10,8 set with (5,6) partition to make sure extra (6th) el is assigned from
		// an eligible group (i.e., not group 5 since it has fewer els) 
//		int[] splitObj = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,
//				25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48};
//		int[] groupDef = {1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,
//				1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5};
//		 2. test 10,8,10,8,10 set with (6,6) partition to make sure els are assigned 
//		 according to group proportions even with smaller than max partition size
//		 (i.e. make sure extra (6th) els come from grps 1,3 or 5)
//		int[] splitObj = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,
//				24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46};
//		int[] groupDef = {1,1,1,1,1,2,2,2,2,3,3,3,3,3,4,4,4,4,5,5,5,5,5,
//				1,1,1,1,1,2,2,2,2,3,3,3,3,3,4,4,4,4,5,5,5,5,5};
		// 3. test 8,10,10,8,10 set with (7,6) partition to make sure els are assigned 
		// according to group proportions even with smaller than max partition size and
		// smaller proportion of 1st grp than other grps even though upper bound on
		// eligible number of els to partition from each group is same (i.e. 3 in this case;
		// see  int[] nObjInGrpPartns in partitionGroups() above).
		// (i.e. make sure extra (6th and 7th) els come from grps 2,3 or 5)
//		int[] splitObj = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,
//				24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46};
//		int[] groupDef = {1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,5,5,5,5,5,
//				1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,5,5,5,5,5};
//		
		//4. test set with single group and (i) odd sub-partition (2,3) (ii) even reg. partition (4,4)
//		int[] splitObj = {1,2,3,4,5,6,7,8};
//		int[] groupDef = {1,1,1,1,1,1,1,2};
//		
		// 5. test 5,4,3,5,2 set with partition (5,10) to make sure that grps of different sizes 
		// with the same upper bound on eligible number of els in a partition (4 for grp 1, 4 for grp 2)
		// but a different number of els (5 in grp 1, only 4 in grp 2) are treated correctly, i.e., that 
		// grp 2 will never have more els than grp 1.  
		// 
//		int[] splitObj = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19};
//		int[] groupDef = {1,1,1,1,1,2,2,2,2,3,3,3,4,4,4,4,4,5,5};
		
		// 6. test regular split-half resampling (random and deterministic) for single group
		// 12 subjects, partition {6,6}
		int[] splitObj = {1,2,3,4,5,6,7,8,9,10,11,12};
		int[] groupDef = {1,1,1,1,1,1,1,1,1,1,1,1};
		
		// 7. test reg. split-half resampling on single grp really small data:
		// 4 subjects, partition {2,2}
//		int[] splitObj = {1,2,3,4};
//		int[] groupDef = {1,1,1,1};
		
		
		int[] partn = {6,6};
		int numSplits = 10;
		boolean random = true;
		
		for (int i = 0; i < 10; ++i) {
		Resampler Res = new Resampler(splitObj, groupDef, partn, numSplits, false);
		try {
			Res.generateSplits(random);
		
		}
		catch (NpairsjException e) {
			e.printStackTrace();
		}
		}
		
		System.exit(0);
	}	
	
	public static void main1(String[] args) {
//	    	int length = 1660;
//	    	int[] splitObj = new int[length];
//	    	int[] grpLabels = new int[length];
//	    	String splitObjFilename = 
//	    		"/home/anita/workspace/PLSwithNPAIRS/localTestData/splitObj.txt";
//	    	String grpLabelFilename = 
//	    		"/home/anita/workspace/PLSwithNPAIRS/localTestData/grpLabels.txt";
//	    	try {
//	    		    
//	    		    java.io.FileInputStream fstreamSplObj = 
//	    		    	new java.io.FileInputStream(splitObjFilename);
//	    		    java.io.FileInputStream fstreamGrpLab = 
//	    		    	new java.io.FileInputStream(grpLabelFilename);
//	    		    // Get the object of DataInputStream
//	    		    java.io.DataInputStream inSplObj = 
//	    		    	new java.io.DataInputStream(fstreamSplObj);
//	    		    java.io.DataInputStream inGrpLab = 
//	    		    	new java.io.DataInputStream(fstreamGrpLab);
//	    		    java.io.BufferedReader brSplObj = 
//	    		        new java.io.BufferedReader(new java.io.InputStreamReader(inSplObj));
//	    		    java.io.BufferedReader brGrpLab = 
//	    		        new java.io.BufferedReader(new java.io.InputStreamReader(inGrpLab));
//	    		    String strLine1;
//	    		    String strLine2;
//	    		    //Read File Line By Line
//	    		    int count = 0;
//	    		    while ((strLine1 = brSplObj.readLine()) != null)   {
//	    		      // Print the content to splitObj array
//	    		    	splitObj[count] = new Integer(strLine1.trim());
//	    		    	++count;
//	    		    }
//	    		    count = 0;
//	    		    while ((strLine2 = brGrpLab.readLine()) != null) {
////	    		    	 Print the content to grpLabel array
//	    		    	grpLabels[count] = new Integer(strLine2.trim());
//	    		    	++count;
//	    		    }
//	    		    //Close the input stream
//	    		    inSplObj.close();
//	    		    inGrpLab.close();
//	    		    
//	    	}
//	    	catch (Exception e) {//Catch exception if any
//	    		    System.err.println("Error: " + e.getMessage());
//	        }
	    	
//		int[] splitObj = {0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,16,16,
//				17,17,18,18,19,19};
//		int[] splitObj = {13,13,0,0,3,3,6,6,4,4,9,9,2,2,5,5,7,7,8,8,19,19,17,17,1,1,10,10,14,14,16,16,11,11,
//				15,15,12,12,18,18};
//		int[] splitObj = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
//				24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
//				47, 48, 49, 50};
		int[] splitObj = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
		int[] grpLabels = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
//		int[] grpLabels = {1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,9,10,10,10,10};
//		int[] grpLabels = {6,6,6,6,3,3,2,2,3,3,10,10,10,10,5,5,7,7,5,5,7,7,2,2,1,1,1,1,4,4,8,8,9,9,9,9,8,8,4,4};
//		int[] grpLabels = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//				0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

		
		
	    	System.out.println("(Resampler.java) Split object array: ");
	    	npairs.io.NpairsjIO.print(splitObj);
	    	System.out.println("(Resampler.java) Grp label array: ");
	    	npairs.io.NpairsjIO.print(grpLabels);
	    	
	    	int numSplits = 20;
//			int numSplits = 1;
//	    	int[] partition = {15, 7};
//	    	int[] partition = {10, 10};
	    	int[] partition = {6, 6}; 
	    	Resampler r = new Resampler(splitObj, grpLabels, partition, numSplits);
	    	boolean random = false;
	    	try {
	    		r.generateSplits(random);
	    	}
	    	catch(NpairsjException e) {
	    		e.printStackTrace();
	    	}
	    		
	   }
	
//	public static void main2 (String[] args) {
//		int[] splitObj = {1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8};
//		int[] grpLabels = {1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0};
//		int[] partition = {2, 2};
//		int numSplits = 10;
//		boolean random = true;
//		Resampler res = new Resampler(splitObj, grpLabels, partition, numSplits);
//		try {
//			int[][][] results = res.generateSplits(random);
//		}
//		catch(NpairsjException e) {
//			e.printStackTrace();
//		}
//	}
}

//		int[] splitObj = new int[10000];
//		int[] grpDef = new int[10000];
//		for (int i = 0; i < 10000; ++i) {
//			splitObj[i] = i%2500; 
//			grpDef[i] = i%10;
//		}
//		int[] partition = {100, 100};
//		int numSplits = 2;
//		Resampler res = new Resampler(splitObj, grpDef, partition, numSplits, false);
//		try {
//			res.generateSplits();
//		}
//		catch (NpairsjException e) { e.printStackTrace(); }
//	}


/**************************************************************************************************/

/** Class CombinationGenerator
 * 
 *  Generates samples of split objects deterministically using the method of
 *  Rosen (p. 286) employed by Vicky. 
 * @author anita
 *
 */
final class CombinationGenerator {
	
	private int total;
	
	private int numLeft;
	
	private int[] a;
	
	private int n;
	
	private int r;
	
	
	/** Constructor.
	 *  REQUIRED: numCombos <= n choose r (not checked in CombinationGenerator code)
	 *  
	 * @param n size of set of objects to be combined
	 * @param r size of each sample set 
	 * @param numCombos total number of combos to generate (must be <= n choose r)
	 */
	protected CombinationGenerator(int n, int r, int maxCombos) {
		this.total = maxCombos;
		this.n = n;
		this.r = r;
		this.a = new int[r];
		reset(); 		
	}
	
	/**
	 * Resets the generator.
	 */

	private void reset() {
		for (int i = 0; i < a.length; i++) {
			a[i] = i;
		}
		numLeft = total;
	}
	
	/**
	 * Returns boolean indicating whether there are more combinations 
	 * to be generated.
	 * @return boolean More combinations left to generate?
	 */
	private boolean hasMore() {
		return numLeft > 0;
	}

	/**
	 * Returns total number of combinations to be generated
	 * @return total number of combos to be generated 
	 *         (<= max possible combos)
	 */
	private int getTotal() {
		return total;
	}
	
	/**
	 * Generate next combination (algorithm from Rosen p. 286)
	 * @return Array of indices in range 0 to n-1(?)
	 */
	private int[] getNext() {

		if (numLeft == total) {
			numLeft -= 1;
			return a;
		}
		
		int i = r - 1;
		while (a[i] == n - r + i) {
			i--;
		}
		
		a[i] = a[i] + 1;
		for (int j = i + 1; j < r; j++) {
			a[j] = a[i] + j - i;
		}
		
		numLeft -= 1;
		return a;
	}

    /**
     * Compute combinations of input elements
     * @param elements to be combined 
     */
    protected HashSet[] generateCombos(int[] elements) {
    	
    	int[] indices;

//    	if (debug) {
//    		System.out.println("Num combos to generate: " + total);
//    	}
//  
    	HashSet[] currSample = new HashSet[total];
    	int count = 0;
    	while (this.hasMore()) {
    		indices = this.getNext();
//    		if (debug) {
//    			System.out.println("Indices: ");
//    			npairs.io.NpairsjIO.print(indices);
//    		}
    		currSample[count] = new HashSet<Integer>(indices.length);
    		for (int i = 0; i < indices.length; ++i) {
    			currSample[count].add(elements[indices[i]]);
    		}
    		++count;
    	}
//      	if (debug) {
//      		System.out.println("currCombos: ");
//      		for (int t = 0; t < total; ++t) {
//      			System.out.println(currSample[t].toString());
//      		}
//      	}
      	return currSample;
    }
    
  
  
}