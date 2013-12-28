/* @(#)QSortAlgorithm.java      1.3.1   26 Oct 2005 James Gosling
 * version 1.3.1 by Ron Ammar
 *
 * Copyright (c) 1994-1996 Sun Microsystems, Inc. All Rights Reserved.
 *
 * Permission to use, copy, modify, and distribute this software
 * and its documentation for NON-COMMERCIAL or COMMERCIAL purposes and
 * without fee is hereby granted. 
 * Please refer to the file http://www.javasoft.com/copy_trademarks.html
 * for further important copyright and trademark information and to
 * http://www.javasoft.com/licensing.html for further important
 * licensing information for the Java (tm) Technology.
 * 
 * SUN MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF
 * THE SOFTWARE, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
 * TO THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 * PARTICULAR PURPOSE, OR NON-INFRINGEMENT. SUN SHALL NOT BE LIABLE FOR
 * ANY DAMAGES SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR
 * DISTRIBUTING THIS SOFTWARE OR ITS DERIVATIVES.
 * 
 * THIS SOFTWARE IS NOT DESIGNED OR INTENDED FOR USE OR RESALE AS ON-LINE
 * CONTROL EQUIPMENT IN HAZARDOUS ENVIRONMENTS REQUIRING FAIL-SAFE
 * PERFORMANCE, SUCH AS IN THE OPERATION OF NUCLEAR FACILITIES, AIRCRAFT
 * NAVIGATION OR COMMUNICATION SYSTEMS, AIR TRAFFIC CONTROL, DIRECT LIFE
 * SUPPORT MACHINES, OR WEAPONS SYSTEMS, IN WHICH THE FAILURE OF THE
 * SOFTWARE COULD LEAD DIRECTLY TO DEATH, PERSONAL INJURY, OR SEVERE
 * PHYSICAL OR ENVIRONMENTAL DAMAGE ("HIGH RISK ACTIVITIES").  SUN
 * SPECIFICALLY DISCLAIMS ANY EXPRESS OR IMPLIED WARRANTY OF FITNESS FOR
 * HIGH RISK ACTIVITIES. */
package npairs.utils;

/** A fast quick sort algorithm.
 *  This version of Fast Quick Sort, is designed to return
 *  the sorted indeces of a double array. That is, if you are given
 *  [3,1,2], the sorted indeces would be [1,2,0] corresponding to
 *  [1,2,3].
 *  @author James Gosling
 *  @author Kevin A. Smith
 *  @author Ron Ammar
 *  @version 1.3.1
 *  History:	October/26/2005 */
public class FastQuickSort {

	/** The sorted indeces of the sorted double array. */
	public int[] sortedIndex;
	
	/** This is a generic version of C.A.R Hoare's Quick Sort 
	 *  algorithm.  This will handle arrays that are already
	 *  sorted, and arrays with duplicate keys.<BR>
	 *  If you think of a one dimensional array as going from
	 *  the lowest index on the left to the highest index on the right
	 *  then the parameters to this function are lowest index or
	 *  left and highest index or right.  The first time you call
	 *  this function it will be with the parameters 0, a.length - 1.
	 *  @param a	an integer array
	 *  @param l	left boundary of array partition
	 *  @param r	right boundary of array partition */
	private void QuickSort(double a[], int l, int r) throws Exception {
		int M= 4;
		int i;
		int j;
		double v;

		if ((r-l) > M) {
			i= (r+l)/2;
			if (a[l] > a[i]) swap(a,l,i);	// Tri-Median Method!
			if (a[l] > a[r]) swap(a,l,r);
			if (a[i] > a[r]) swap(a,i,r);

			j= r-1;
			swap(a,i,j);
			i= l;
			v= a[j];
			
			for(;;) {
				while(a[++i] < v);
				while(a[--j] > v);
				if (j < i) break;
				swap (a,i,j);
			}
			
			swap(a,i,r-1);
			QuickSort(a,l,j);
			QuickSort(a,i+1,r);
		}
	}

	
	private void swap(double a[], int i, int j) {
		double T;
		
		T= a[i]; 
		a[i]= a[j];
		a[j]= T;

		swapIndex(i, j); //keep track of changing indeces
	}

    	
	private void InsertionSort(double a[], int lo0, int hi0) 
		throws Exception {
		/* This method parallels all array manipulations on the
		 * index array. */
		int i;
		int j;
		double v;
		int tempIndex;
		
		for (i= lo0 + 1; i <= hi0; i++) {
			v= a[i];
			tempIndex= sortedIndex[i];			
			j= i;
			while ((j > lo0) && (a[j-1] > v)) {
				a[j] = a[j-1];
				sortedIndex[j]= sortedIndex[j-1];
				j--;
			}
			a[j] = v;
			sortedIndex[j]= tempIndex;
		}
	}


	/** Fill the index array with ordered indeces to be sorted.
	 *  @param length	the array length = # of indeces */
	private void fillIndex(int length) {
		for(int i= 0; i != length; ++i) {
			sortedIndex[i]= i;
		}
	}

	/** Swap the indeces of two array elements. Only do this when
	 *  the array elements are swapped for a sort (as done in the
	 *  "swap" method).
	 *  @param i	index of one element
	 *  @param j	index of second element */
	private void swapIndex(int i, int j) {
		int temp= sortedIndex[i];
		sortedIndex[i]= sortedIndex[j];
		sortedIndex[j]= temp;
	}


	// PUBLIC METHODS //
	
	/** Sorts the given array.
	 *  @param a	the input double array to be sorted */
	public void sort(double a[]) {
		try {
			//Fill the index array according to the double array length
			sortedIndex= new int[a.length];
			fillIndex(a.length);
	
			QuickSort(a, 0, a.length - 1);
			InsertionSort(a, 0, a.length - 1);
		} catch (Exception e) {
			System.err.println("FastQuickSort error: " + e);
		}
	}

	
	public void printArray(double[] a) {
		String result= "";

		result += "[";
		for(int i= 0; i!= a.length; ++i) {
			result += a[i];
			if (i != a.length - 1)
				result += ", ";
		}
		result += "]";
		System.out.println(result);
	}


	public void printIndex() {
		String result= "";

		result += "[";
		for(int i= 0; i != sortedIndex.length; ++i) {
			result += sortedIndex[i];
			if (i != sortedIndex.length - 1)
				result += ", ";
		}
		result += "]";
		System.out.println(result);
	}
	
	
	/** A sample simple test of the fast quick sort. */
	public static void simpleTest() throws Exception {
		FastQuickSort f= new FastQuickSort();
		double[] d= {10.0,4.0,1.0,3.0,2.0,1.3,5.6,6.1,7.999234,9.231,8.0};

		System.out.println("Input:");
		f.printArray(d);
		System.out.println("Unsorted Index:" +
				System.getProperty("line.separator") +
				"[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]");
		System.out.println("Expected Sort:" + 
				System.getProperty("line.separator") +
		"[1.0, 1.3, 2.0, 3.0, 4.0, 5.6, 6.1, 7.999234, 8.0, 9.231, 10.0]");
		f.sort(d);
		System.out.println("Sorted:");
		f.printArray(d);
		System.out.println("Sorted Index:");
		f.printIndex();

		/* NOTE: Although not demonstrated here, the sort method has
		 * been shown to sort, without error, empty arrays, single
		 * element arraysand arrays with negative numbers. All are
		 * arrays of type double. */
	}

}
