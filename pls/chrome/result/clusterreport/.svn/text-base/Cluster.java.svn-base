package pls.chrome.result.clusterreport;

import java.util.HashSet;

/**
 * A simple class to represent a cluster as found in a brain volume.
 * 
 * i.e. An adjacent grouping of statistically significant voxels.
 */
@SuppressWarnings("serial")
public class Cluster extends HashSet<Voxel> {
	private int mId;
	
	public Voxel mPeakVoxel = null;
	
	public Cluster(int id)
	{
		mId = id;
	}
	
	/**
	 * For use in hash set, this class overrides hashCode so that its hash code
	 * does not change when adding new elements to it.
	 */
	public int hashCode()
	{
		return mId;
	}
	
	public boolean equals(Cluster o)
	{
		return mId == o.mId;
	}
	
	/**
	 * Overrides the add method so that the voxel being added to this cluster
	 * is aware of which cluster it belongs to.
	 */
	public boolean add(Voxel o)
	{
		o.cluster = this;
		
		if (mPeakVoxel == null || Math.abs(mPeakVoxel.value) < Math.abs(o.value) )
		{
			mPeakVoxel = o;
		}
		
		return super.add(o);
	}
	
	public boolean addAll(Cluster otherCluster)
	{
		for (Voxel v : otherCluster)
		{
			if (!add(v) )
				return false;
		}
		
		return true;
	}
}
