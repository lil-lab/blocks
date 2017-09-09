using System;
using System.Collections.Generic;

public class CachedTrajectory {
	private List<Entry> entries;

	public CachedTrajectory () {
		this.entries = new List<Entry> ();
	}

	public bool isExist(Episode eps, int startIndex, int endIndex) {
		foreach(Entry entry in this.entries) {
			if(entry.equal(eps, startIndex, endIndex)) {
				return true;
			}
		}
		return false;
	}

	public void addTrajectory(Episode eps, int startIndex, int endIndex, TrajectoryResult trajResult) {
		this.entries.Add (new Entry (eps, startIndex, endIndex, trajResult));
	}

	public TrajectoryResult getTrajectory(Episode eps, int startIndex, int endIndex) {
		foreach(Entry entry in this.entries) {
			if(entry.equal(eps, startIndex, endIndex)) {
				return entry.getTrajectoryResult ();
			}
		}
		throw new ApplicationException ("Trajectory not found");
	}

	public int size() {
		return this.entries.Count;
	}

	public void saveTrajectory() {

		System.IO.StreamWriter file = new System.IO.StreamWriter ("Assets/cached_trajectory.txt");
		foreach (Entry entry in this.entries) {
			List<int> traj = entry.getTrajectoryResult ().getTrajectory ();
			string trajS = "";
			for (int i = 0; i < traj.Count; i++) {
				if (i < traj.Count - 1) {
					trajS = trajS + traj [i] + ",";
				} else {
					trajS = trajS + traj [i];
				}
			}
			file.WriteLine (trajS);
		}

		file.Flush ();
		file.Close ();
	}
}

public class Entry {

	private Episode eps;
	private int startIndex;
	private int endIndex;
	private TrajectoryResult trajResult;

	public Entry(Episode eps, int startIndex, int endIndex, TrajectoryResult trajResult) {
		this.eps = eps;
		this.startIndex = startIndex;
		this.endIndex = endIndex;
		this.trajResult = trajResult;
	}

	public bool equal(Episode eps, int startIndex, int endIndex) {
		if (this.eps == eps && this.startIndex == startIndex && this.endIndex == endIndex) {
			return true;
		} else {
			return false;
		}
	}

	public TrajectoryResult getTrajectoryResult() {
		return this.trajResult;
	}
}

