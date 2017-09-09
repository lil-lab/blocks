using System;

public class SynchronizedInt {
	private int val;

	public SynchronizedInt () {
		this.val = 0;
	}
		
	public SynchronizedInt (int initVal) {
		this.val = initVal;
	}

	public int getVal() {
		lock (this) {
			return this.val;
		}
	}

	public void reset() {
		this.setVal(0);
	}

	public void setVal(int newVal) {
		lock (this) {
			this.val = newVal;
		}
	}

	public void increment() {
		lock (this) {
			this.val++;
		}
	}
}

