package pls.chrome.result.clusterreport;

//import java.util.HashMap;
import java.util.HashSet;
import java.util.Vector;

import pls.chrome.result.model.BrainData;
import pls.chrome.result.model.ResultModel;

public class ClusterCalculator {
	private BrainData mBrainData = null;
	private Vector<HashSet<Cluster> > mClusters = null;
	private Voxel[][][] mVoxels = null;
	private double mThreshold = 0;
	private int mCounter = 0;
	
	
	public ClusterCalculator(ResultModel model)
	{
		mBrainData = model.getBrainData();
		
		mThreshold = mBrainData.getColourScaleModel().getColourScale()[2];
		
		mClusters = new Vector<HashSet<Cluster> >();
		
//		HashMap<Integer, double[]> data = mBrainData.getData();
		
		int numLags = model.getWindowSize();
		
		int depth = mBrainData.getNumSlices(BrainData.AXIAL);
		int width = mBrainData.getWidth(BrainData.AXIAL);
		int height = mBrainData.getHeight(BrainData.AXIAL);
		
		mVoxels = new Voxel[depth][height][width];
		
		// Iterate through each lag
		for (int lagNum = 0; lagNum < numLags; ++lagNum)
		{
			HashSet<Cluster> currClusterSet = new HashSet<Cluster>();
			// Iterate through each slice
			for (int z = 0; z < depth; ++z)
			{	
				// Iterate through each voxel on current slice
				for (int y = 0; y < height; ++y)
				{
					for (int x = 0; x < width; ++x)
					{
						double value = mBrainData.getValue3D(x, y, z, lagNum);
						
						// Add 1 to y and z because we are trying to stay
						// consistent with the display from Matlab (which uses
						// 1-based indexing).
						Voxel currentVoxel = new Voxel(x + 1, y + 1, z + 1, value, null); 
						
						// If the value is beyond threshold
						if (Math.abs(currentVoxel.value) > Math.abs(mThreshold) )
						{
							
							// Try to group this voxel with neighbouring voxels
							if (x > 0)
							{
								checkVoxel(currentVoxel, mVoxels[z][y][x-1], currClusterSet);
							}
							
							if (y > 0)
							{
								checkVoxel(currentVoxel, mVoxels[z][y-1][x], currClusterSet);
							}
							
							if (z > 0)
							{
								checkVoxel(currentVoxel, mVoxels[z-1][y][x], currClusterSet);
							}
							
							// If no proper neighbours were found, create a
							// new cluster for this voxel.
							if (currentVoxel.cluster == null)
							{
								currentVoxel.cluster = new Cluster(mCounter++);
								currentVoxel.cluster.add(currentVoxel);
								currClusterSet.add(currentVoxel.cluster);
							}
						}
						
						mVoxels[z][y][x] = currentVoxel;
					}
				}
			}
			
			mClusters.add(currClusterSet);
		}
	}
		
	private void checkVoxel(Voxel currentVoxel, Voxel potentialVoxel, HashSet<Cluster> currClusterSet)
	{
		// If these voxels are on the same side of the threshold
		if ( (potentialVoxel.value > mThreshold && currentVoxel.value > mThreshold) ||
				(potentialVoxel.value < -mThreshold && currentVoxel.value < -mThreshold) )
		{
			// If the current voxel's cluster is null, then we simply assign
			// it the potential voxel's cluster.
			if (currentVoxel.cluster == null)
			{
				potentialVoxel.cluster.add(currentVoxel);
			}
			// Otherwise we merge the clusters together.
			else
			{
				Cluster potentialCluster = potentialVoxel.cluster;
				
				// But only if they aren't already part of the same cluster
				if (potentialCluster != currentVoxel.cluster)
				{
					
					currentVoxel.cluster.addAll(potentialCluster);
					
					boolean removed = currClusterSet.remove(potentialCluster);
					
					assert(removed);
				}
			}
		}
	}
	
	/**
	 * Returns the collection of clusters
	 */
	public Vector<HashSet<Cluster> > getClusters()
	{
		return mClusters;
	}
	
	/**
	 * Very basic toString method.  Use for debugging only. 
	 */
	public String toString()
	{
		String returnValue = "";
		int i = 1;
		
		for (HashSet<Cluster> hset : mClusters)
		{
			for (Cluster c : hset)
			{
				if (c.size() >= 5)
				{
					returnValue += "Cluster #" + i + "\n";
					returnValue += "Number of voxels: " + c.size() + "\n";
					++i;
				}
			}
		}
		
		return returnValue;
	}
}
