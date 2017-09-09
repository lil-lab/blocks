using System;
using System.Collections.Generic;

/** List of environments (locations of cubes on the board) where 
* next environment in the list proceeds the first environment. */
public class Episode {

	private List<ObjectSystem> systems;
	private float blockSize;

	public Episode (List<ObjectSystem> systems, float blockSize) {
		this.systems = systems;
		this.blockSize = blockSize;
	}

	public ObjectSystem getFirstEnvironment() {
		return this.systems [0];
	}

	public ObjectSystem getEnvironmentByIndex(int index) {
		return this.systems [index];
	}
		
	public int numEnvironment() {
		return this.systems.Count;
	}

	public float getBlockSize() {
		return this.blockSize;
	}
}

