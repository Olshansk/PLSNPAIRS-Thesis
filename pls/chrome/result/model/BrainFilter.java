package pls.chrome.result.model;

import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

public class BrainFilter {
	private TreeMap<String, TreeSet<Integer> > mFilters = new TreeMap<String, TreeSet<Integer> >();
	private TreeMap<String, Boolean> mEnabled = new TreeMap<String, Boolean>();
	
//	private TreeSet<Integer> mCoords = new TreeSet<Integer>();
//	private TreeSet<Integer> mFilteredCoords = new TreeSet<Integer>();
	
	public Set<String> filterNames() {
		return mFilters.keySet();
	}

	public TreeSet<Integer> getFilter(String filterName){
		return mFilters.get(filterName);
	}

	public void toggleFilterEnabled(String filterName) {
		mEnabled.put(filterName, !mEnabled.get(filterName) );
		
//		refreshFilteredCoords();
	}
	
	public boolean isEnabled(String filterName) {
		return mEnabled.get(filterName);
	}
	
//	public void setCoords(int[] coords) {
//		mCoords = new TreeSet<Integer>();
//		for (int i : coords) {
//			mCoords.add(i);
//		}
//		
//		refreshFilteredCoords();
//	}
	
//	public TreeSet<Integer> getCoords() {
//		return mCoords;
//	}
	
	public void addFilter(String filterName, TreeSet<Integer> newFilter) {
		mFilters.put(filterName, newFilter);
		mEnabled.put(filterName, true);
		
//		refreshFilteredCoords();
	}
	
	public void removeFilter(String filterName) {
		if (mFilters.containsKey(filterName) ) {
			mFilters.remove(filterName);
			mEnabled.remove(filterName);
		}

//		refreshFilteredCoords();
	}
	
//	private void refreshFilteredCoords() {
////		mFilteredCoords = new TreeSet<Integer>(mCoords);
//		mFilteredCoords = null;
//		
//		for (String s : mFilters.keySet() ) {
//			
//			if (mEnabled.get(s) ) {
//				TreeSet<Integer> filter = mFilters.get(s);
//				if (mFilteredCoords == null) {
//					mFilteredCoords = new TreeSet<Integer>(filter);
//				} else {
//					mFilteredCoords.retainAll(filter);
//				}
//			}
//		}
//	}

	/**
	 *
	 * @return list of coordinates constructed from intersection of all
	 * coordinates in filtered sets. Empty set if no intersection or no filters.
	 */
	public TreeSet<Integer> getFilteredCoords() {
		TreeSet<Integer> filteredCoords = null;
		
		for (String s : mFilters.keySet() ) {
			
			if (mEnabled.get(s) ) {
				TreeSet<Integer> filter = mFilters.get(s);

				if (filteredCoords == null) {
					filteredCoords = new TreeSet<Integer>(filter);
				} else {
					filteredCoords.retainAll(filter);
				}
			}
		}
		//
		if (filteredCoords == null) 
			filteredCoords = new TreeSet<Integer>();
		if (filteredCoords == null) 
			filteredCoords = new TreeSet<Integer>();
		return filteredCoords;
	}

	/**
	 * @param coords to be filtered.
	 * @return coords The coordinates resulting from the intersection of the
	 * passed in coordinates and all filtered coordinate sets.
	 */
	public TreeSet<Integer> getFilteredCoords(TreeSet<Integer> coords) {
		TreeSet<Integer> filteredCoords = new TreeSet<Integer>(coords);
		
		for (String s : mFilters.keySet() ) {
			
			if (mEnabled.get(s) ) {
				TreeSet<Integer> filter = mFilters.get(s);
				filteredCoords.retainAll(filter);
			}
		}
		
		return filteredCoords;
	}
}
