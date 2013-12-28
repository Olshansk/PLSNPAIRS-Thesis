package pls.chrome.result.clusterreport;

/**
 * A simple class to represent a voxel in a cluster.  Contains the position
 * of the voxel, its value and the cluster it belongs to (if any).
 */
public class Voxel {
	private int x;
	private int y;
	private int z;
	
	public double value;
	
	public Cluster cluster;
	
	public Voxel(int x, int y, int z, double value, Cluster cluster)
	{
		this.x = x;
		this.y = y;
		this.z = z;
		this.value = value;
		this.cluster = cluster;
	}
	
	public int getX() {
		return x;
	}
	
	public void setX(int x) {
		this.x = x;
	}
	
	public int getY() {
		return y;
	}
	
	public void setY(int y) {
		this.y = y;
	}
	
	public int getZ() {
		return z;
	}
	
	public void setZ(int z) {
		this.z = z;
	}
}
