using UnityEngine;
using System;
using System.Collections.Generic;

// Min heap implementation
class MinHeap<T> where T : IComparable  {

	// Element at index i has to children
	// at index 2i + 1 and 2i + 2
	List<T> data;

	public MinHeap() {
		this.data = new List<T> ();
	}

	// Returns the minimum value element without removing it
	public T peek() {

		if (data.Count == 0) {
			throw new ApplicationException ("List is empty");
		}

		return data [0];
	}

	// Returns size of the heap
	public int size() {
		return this.data.Count;
	}

	// Removes and returns the minimum value element
	public T extractMin() {

		T min = this.data [0];
		this.data [0] = this.data [this.data.Count - 1];
		this.data.RemoveAt (this.data.Count - 1);
		
		this.heapifyDownward (0);

		return min;
	}

	public void heapifyDownward(int i) {
		
		while(true) {
			// Check if i is smaller than or equal to both its child
			// and if not then swap and continue

			if (i >= this.data.Count) {
				break;
			}

			int j = i;
			T node = this.data [i];

			T child1 = default(T);
			bool c1 = true;
			if (2 * i + 1 <= this.data.Count - 1) {
				child1 = this.data[2 * i + 1];
				c1 = false;
			} 

			T child2 = default(T);
			bool c2 = true;
			if (2 * i + 2 <= this.data.Count - 1) {
				child2 = this.data[2 * i + 2];
				c2 = false;
			}

			if (c1 || node.CompareTo (child1) <= 0) {
				if (c2 || node.CompareTo (child2) <= 0) {
					// Parent smaller than or equal to both children
					break;
				} else {
					i = 2 * i + 2; //child2 is smallest
				}
			} else {
				if (c2 || child1.CompareTo (child2) <= 0) {
					i = 2 * i + 1; //child1 is smallest
				} else {
					i = 2 * i + 2; //child2 is smallest
				}
			}

			// Swap i and j
			this.swap(i, j);
		}
	}

	// Insert a element to heap
	public void insert(T o) {
		data.Add (o);
		this.heapifyUpward (data.Count - 1);
	}

	// Delete a element from the heap
	public void delete(T o) {
		
		int i = data.IndexOf (o);

		// Edge case
		if (i == this.data.Count - 1) {
			this.data.RemoveAt (this.data.Count - 1);
			return;
		}

		this.data [i] = this.data [this.data.Count - 1];
		this.data.RemoveAt (this.data.Count - 1);

		// Check the value of the new data[ix]
		T val = this.data[i];

		T child1 = default(T);
		bool c1 = true;
		if (2 * i + 1 <= this.data.Count - 1) {
			child1 = this.data[2 * i + 1];
			c1 = false;
		} 

		T child2 = default(T);
		bool c2 = true;
		if (2 * i + 2 <= this.data.Count - 1) {
			child2 = this.data[2 * i + 2];
			c2 = false;
		}

		// If val is smaller than both child then heap upward.
		// If val is strictly greater than any one of the child then by
		// extension it is greater than all parents. In this case
		// heap downard.

		if (c1 || val.CompareTo (child1) <= 0) {
			if (c2 || val.CompareTo (child2) <= 0) {
				// val smaller than or equal to both children
				this.heapifyUpward (i);
			} else {
				// val strictly greater than one child
				this.heapifyDownward(i);
			}
		} else {
			// val strictly greater than one child
			this.heapifyDownward(i);
		}
	}

	private void swap(int i, int j) {
		T temp = this.data [i];
		this.data [i] = this.data [j];
		this.data [j] = temp;
	}

	// Called after insert or extractMin to 
	// ensure that heap property holds.
	private void heapifyUpward(int i) {
		
		int j = i;

		while (j > 0) {

			T node = this.data [j];
			int pix = (int)((j - 1) / 2.0);
			T parent = this.data[pix];

			if (parent.CompareTo (node) > 0) { 
				//parent of node has larger value than node
				this.swap(j, pix);
				j = pix;
			} else {
				break;
			}
		}
	}

	public List<T> getData() {
		return this.data;
	}
}