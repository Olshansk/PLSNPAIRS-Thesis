package pls.chrome.result.clusterreport;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Vector;

import javax.swing.JTable;
import javax.swing.RowSorter;
import javax.swing.SortOrder;
import javax.swing.table.DefaultTableModel;
import javax.swing.table.TableRowSorter;

import pls.chrome.result.model.ResultModel;

public class ClusterReportMaker {
	Vector<HashSet<Cluster> > mClusters = null;
	
	private ResultModel mResultModel = null;
	
	int mMinimumSize = 0;
	
	public ClusterReportMaker(ResultModel model, Vector<HashSet<Cluster> > clusters, int minimumSize)
	{
		mClusters = clusters;
		mResultModel = model;
		mMinimumSize = minimumSize;
	}
	
	public JTable generateReport()
	{
		int clusterNumber = 1;
		
		DefaultTableModel model = new DefaultTableModel();
		model.addColumn("Cluster #");
		model.addColumn("Lag #");
		model.addColumn("XYZ");
		model.addColumn("XYZ (mm)");
		model.addColumn("Peak Value");
		model.addColumn("Cluster Size (voxels)");
		
		JTable table = new JTable(model);
		
		// Create a custom sorter
		TableRowSorter<DefaultTableModel> sorter = new TableRowSorter<DefaultTableModel>(model);
		
		// Set a sorter for each column of importance
		sorter.setComparator(0, new StandardIntegerSorter() );
		sorter.setComparator(1, new StandardIntegerSorter() );
		sorter.setComparator(4, new PeakValueSorter() );
		sorter.setComparator(5, new StandardIntegerSorter() );
		
		ArrayList <RowSorter.SortKey> sortKeys = new ArrayList<RowSorter.SortKey>();
		sortKeys.add(new RowSorter.SortKey(1, SortOrder.ASCENDING));
		sortKeys.add(new RowSorter.SortKey(4, SortOrder.ASCENDING));
		sortKeys.add(new RowSorter.SortKey(5, SortOrder.ASCENDING));
		sortKeys.add(new RowSorter.SortKey(2, SortOrder.ASCENDING));
		sortKeys.add(new RowSorter.SortKey(3, SortOrder.ASCENDING));
		sortKeys.add(new RowSorter.SortKey(0, SortOrder.ASCENDING));
		sorter.setSortKeys(sortKeys);
	
		table.setRowSorter(sorter);
		table.setAutoCreateColumnsFromModel(false);
		
		int i = 0;
		
		for (HashSet<Cluster> hset : mClusters)
		{
			for (Cluster c : hset)
			{
				if (c.size() >= mMinimumSize)
				{
					int[] origin = mResultModel.getBrainData().getOrigin();
					double[] size = mResultModel.getBrainData().getVoxelSize();
					
					Voxel peakVoxel = c.mPeakVoxel;
					
					String XZY = peakVoxel.getX() + ", " + peakVoxel.getY() + ", " + peakVoxel.getZ(); 
					String mm = (peakVoxel.getX() - origin[0]) * size[0] + ", " + (peakVoxel.getY() - origin[1]) * size[1] + ", " +  (peakVoxel.getZ() - origin[2]) * size[2];   
					
					model.addRow(new Object[]{clusterNumber++,
							i,
							XZY,
							mm,
							peakVoxel.value,
							c.size()});
				}
			}
			
			++i;
		}
		
		return table;
	}
    
    public class StandardIntegerSorter implements Comparator<Integer> {
		public int compare(Integer d1, Integer d2) {
			if (d1 < d2) {
				return -1;
			}
			else if (d1 > d2) {
				return 1;
			}
			else {
				return 0;
			}
		}
    	
    }
    
    // This comparator is used to sort a cluster report by its peak value
    // magnitude, grouping positives and negatives together.
    public class PeakValueSorter implements Comparator<Double> {

		public int compare(Double d1, Double d2) {
			// Handle null
			if (d1 == null) {
				if (d2 == null) {
					return 0;
				}
				else {
					return 1;
				}
			}
			else if (d2 == null) {
				return -1;
			}
			
			if (d1 < 0) {
				if (d2 < 0) {
					if (d1 < d2) {
						return -1;
					}
					else if (d1 > d2){
						return 1;
					}
					else {
						return 0;
					}
				}
				else {
					return 1;
				}
			}
			else if (d2 > 0) {
				if (d1 < d2) {
					return 1;
				}
				else if (d1 > d2) {
					return -1;
				}
				else {
					return 0;
				}
			}
			else {
				return -1;
			}
		}
    }
}