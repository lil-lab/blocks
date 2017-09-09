using System;
using System.Collections.Generic;

public class DataPoint {
	
	// Instruction for this datapoint
	private String instruction;
	private Type type;

	// Starting frame
	private int start;

	// End frame ( > start)
	private int end;

	// User ID that created this datapoint
	private String userID;

	// Episode associated with this datapoint
	private Episode episode;

	// Decoration for this datapoint. This tells whether the cubes
	// have a logo, are blank or a digit for this datapoint.
	private Decoration decoration;

	public DataPoint (String instruction, Type type, int start, int end, 
		String userID, Episode episode, Decoration decoration) {

		this.instruction = instruction;
		this.type = type;
		this.start = start;
		this.end = end;
		this.userID = userID;
		this.episode = episode;
		this.decoration = decoration;
	}

	public  String getInstruction() {
		return this.instruction;
	}

	public int getStartFrame() {
		return this.start;
	}

	public int getEndFrame() {
		return this.end;
	}

	public String getUserID() {
		return this.userID;
	}

	public Episode getEpisode() {
		return this.episode;
	}

	public Type getType() {
		return this.type;
	}

	public Decoration getDecoration() {
		return this.decoration;
	}

	public List<int> blocksMoved() {

		List<Point3D> systemi = this.episode.getEnvironmentByIndex (this.start).getObjectLocations();
		List<Point3D> systemf = this.episode.getEnvironmentByIndex (this.end).getObjectLocations();

		List<int> ids = new List<int> ();

		for (int i = 0; i < systemi.Count; i++) {
			Point3D p1 = systemi [i];
			Point3D p2 = systemf [i];

			if (p1.calc2DL2Distance (p2) > 0.001) {
				ids.Add (i);
			}
		}
	
		return ids;
	}
}