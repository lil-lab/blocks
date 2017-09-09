using System;
using System.Collections;
using UnityEngine;
using System.Collections.Generic;

public class ObjectSystem {

	private readonly double sizeBlock;
	private readonly double stepSize;
	private readonly List<Point3D> objectLocations;

	private bool disableTrajectory;
	private List<int> dummyList;

	public ObjectSystem(double sizeBlock, List<Point3D> objectLocations) {
		this.sizeBlock = sizeBlock;
		this.objectLocations = objectLocations;
		this.disableTrajectory = false;
		this.dummyList = new List<int> ();
		this.stepSize = 0.1f;
	}

	public double getSizeOfBlock() {
		return this.sizeBlock;
	}

	public List<Point3D> getObjectLocations() {
		return this.objectLocations; 
	}

	private Point3D applyAction(Point3D pt, int actionID) {

		if (actionID == 0) {
			return new Point3D (pt.getX () + this.stepSize, pt.getY (), pt.getZ ());
		} else if (actionID == 1) {
			return new Point3D (pt.getX () - this.stepSize, pt.getY (), pt.getZ ());
		} else if (actionID == 2) {
			return new Point3D (pt.getX (), pt.getY (), pt.getZ () - this.stepSize);
		} else if (actionID == 3) {
			return new Point3D (pt.getX (), pt.getY (), pt.getZ () + this.stepSize);
		} else {
			throw new ApplicationException ("action id has to be in [0, 3]");
		}
	}
		
	// Check if the point is inside pt
	private bool checkPointInside(Point3D pt, Point3D cube) {

		double tol = 0.002f;
		double half = this.sizeBlock / 2.0;
		double xmin = cube.getX () - half;
		double xmax = cube.getX () + half;
		double zmin = cube.getZ () - half;
		double zmax = cube.getZ () + half;

		if (xmin - tol <= pt.getX () && pt.getX () <= xmax + tol &&
		    zmin - tol <= pt.getZ () && pt.getZ () <= zmax + tol) {
			return true;
		} else {
			return false;
		}
	}

	// Check if cube1 and cube2 collide
	private bool checkCollision(Point3D cube1, Point3D cube2) {

		double half = this.sizeBlock / 2.0;
		Point3D a = new Point3D (cube1.getX () + half, cube1.getY (), cube1.getZ () + half);
		Point3D b = new Point3D (cube1.getX () + half, cube1.getY (), cube1.getZ () - half);
		Point3D c = new Point3D (cube1.getX () - half, cube1.getY (), cube1.getZ () + half);
		Point3D d = new Point3D (cube1.getX () - half, cube1.getY (), cube1.getZ () - half);

		return this.checkPointInside (a, cube2) || this.checkPointInside (b, cube2)
		|| this.checkPointInside (c, cube2) || this.checkPointInside (d, cube2);
	}

	// Checks if blockId at position cube collides with other block
	private int checkCollision(Point3D cube, int blockId) {

		List<Point3D> objLocs = this.getObjectLocations ();

		int ix = 0;
		foreach (Point3D obj in objLocs) {
			if (ix == blockId) {
				ix++;
				continue;
			} else {
				ix++;
			}

			if (this.checkCollision (cube, obj)) {
				return 0;
			}
		}

		return 1;
	}

	private bool checkTrajectoryForCollision(List<int> trajectory, int[,] collides) {

		if (trajectory.Count == 1)
			return true;

		int blockToMove = (int)(trajectory [0] / 4.0);

		Debug.Log ("Block to move " + blockToMove + " found " + this.objectLocations.Count + " trajectory " + trajectory[0]);
		Point3D origin = this.objectLocations [blockToMove];
		Point3D pt = origin;
		Pos center = Pos.of (40, 40);

		foreach(int i in trajectory) {

			if (i == 80)
				break;
			
			pt = this.applyAction(pt, i % 4);

			Pos pos = this.convToPos(pt, center, origin, stepSize);

			int newCollisionValue = this.checkCollision (pt, blockToMove);

			if (newCollisionValue != 0) {
				Debug.Log ("Safe to move to point (" + pt.getX () + ", " + pt.getY () + ", " + pt.getZ () + ")");

				double minD = Double.PositiveInfinity;
				List<Point3D> objLocs = this.getObjectLocations ();

				int ix = 0; int j = 0;
				foreach (Point3D obj in objLocs) {
					if (ix == blockToMove) {
						continue;
					}

					double newDist = pt.calc2DL2Distance (obj);
					if (newDist < minD) {
						j = ix;
						minD = newDist;
					}
					ix++;
					j++;
				}

				Debug.Log ("Min distance " + minD + " index is " + j + " block size " + this.sizeBlock);
				GameObject go = GameObject.Find ("Cube_16");

				if (go != null) {
					Vector3 v16 = go.transform.position;
					Point3D p = new Point3D (v16.x, v16.y, v16.z);
					double dist16 = pt.calc2DL2Distance (p);
					Debug.Log ("Dist16 " + dist16 + " Cube 16 is at (" + v16.x + ", " + v16.y + "," + v16.z + ")");
				}
			}

			if (newCollisionValue != collides[pos.getX(), pos.getY()]) {
				Debug.Log ("Values mismatch new " + newCollisionValue + " vs. old " + collides [pos.getX (), pos.getY ()]);
				Debug.Log ("Pt is (" + pt.getX () + " , " + pt.getZ () + ") Trajectory is not safe");
				return false;
			}
		}

		Debug.Log ("Trajectory is safe");

		return true;
	}

	// Uses A* algorithm to find the optimal path. General as in works with arbitrary number of objects.
	public int findGeneralShortestPathAStar(ObjectSystem final) {

		// Find the index of the object that is different between the two environments
		// there should be exactly one
		List<int> diffBlocks = new List<int> ();
		for (int k = 0; k < this.objectLocations.Count; k++) {

			Point3D o1 = this.objectLocations [k];
			Point3D o2 = final.objectLocations [k];

			if (o1.calc2DL2Distance (o2) > 0.01f) {
				diffBlocks.Add (k);
			}
		}

		List<int> totalTrajectory = new List<int> ();

		foreach (int blockId in diffBlocks) {

			Point3D obj = this.objectLocations [blockId];
			Point3D objf = final.objectLocations [blockId];

			const int maxTrajSize = 40;
			const double tol = 0.001;

			bool[,] closedSet = new bool[2 * maxTrajSize + 1, 2 * maxTrajSize + 1];
			bool[,] openSet = new bool[2 * maxTrajSize + 1, 2 * maxTrajSize + 1];
			float[,] gScore = new float[2 * maxTrajSize + 1, 2 * maxTrajSize + 1];
			float[,] fScore = new float[2 * maxTrajSize + 1, 2 * maxTrajSize + 1];
			int[,] action = new int[2 * maxTrajSize + 1, 2 * maxTrajSize + 1];
			Pos[,] parent = new Pos[2 * maxTrajSize + 1, 2 * maxTrajSize + 1];
			Node[,] pointer = new Node[2 * maxTrajSize + 1, 2 * maxTrajSize + 1];
			int[,] trajSize = new int[2 * maxTrajSize + 1, 2 * maxTrajSize + 1];
			//Collide: -1 not evaluate, 0 evaluated and not safe, 1 evaluated and safe 
			int[,] collides = new int[2 * maxTrajSize + 1, 2 * maxTrajSize + 1];

			for (int j = 0; j < 2 * maxTrajSize + 1; j++) {
				for (int k = 0; k < 2 * maxTrajSize + 1; k++) {
					closedSet [j, k] = false;
					openSet [j, k] = false;
					gScore [j, k] = float.PositiveInfinity;
					fScore [j, k] = float.PositiveInfinity;
					action [j, k] = -1;
					parent [j, k] = null;
					pointer [j, k] = null;
					trajSize [j, k] = -1;
					collides [j, k] = -1;
				}
			}

			MinHeap<Node> priorityQueue = new MinHeap<Node> ();

			// Initialize with the starting state
			openSet [maxTrajSize, maxTrajSize] = true;
			gScore [maxTrajSize, maxTrajSize] = 0;
			fScore [maxTrajSize, maxTrajSize] = (float)obj.calc2DL2Distance (objf);

			Pos origin = Pos.of (maxTrajSize, maxTrajSize);
			Node originNode = new Node (obj, origin, fScore [maxTrajSize, maxTrajSize]);

			pointer [maxTrajSize, maxTrajSize] = originNode;
			trajSize [maxTrajSize, maxTrajSize] = 0;
			priorityQueue.insert (originNode);

			// Output values
			double minAchieveDist = Double.PositiveInfinity;
			Pos goal = null;
			bool reached = false;

			while (priorityQueue.size () > 0) {

				// Get the lowest cost point
				Node lowestCostPt = priorityQueue.extractMin ();
				Pos pos = lowestCostPt.getPos ();
				Point3D pt = lowestCostPt.getPoint ();

				// Check if this is the closest distance reached to goal
				double goalDist = pt.calc2DL2Distance (objf);
				if (goalDist < minAchieveDist) {
					goal = pos;
					minAchieveDist = goalDist;
				}

				// Check if it is close enough to the goal (with margin tol)
				if (goalDist < tol) {
					goal = pos;
					reached = true;
					minAchieveDist = goalDist;
					Debug.Log ("Reached the goal state. goal dist " + goalDist + " and tol " + tol);
					break;
				}

				if (trajSize [pos.getX (), pos.getY ()] >= maxTrajSize) {
					continue;
				}

				// Remove it from openset
				openSet [pos.getX (), pos.getY ()] = false;

				// Add it to closedSet
				closedSet [pos.getX (), pos.getY ()] = true;

				// Find the neighbors of this point and add to score
				// removing those which lead to collision
				for (int actionIx = 0; actionIx < 4; actionIx++) {

					// Apply the action to generate the new Point3D
					Pos newPos = this.applyAction (pos, actionIx);

					if (newPos == null) {
						continue;
					}

					int x = newPos.getX ();
					int y = newPos.getY ();

					// Check if newPt is in closed set or has been shown to collide
					// before. If yes then continue
					if (closedSet [x, y] || collides [x, y] == 0) {
						continue;
					}

					Point3D newPt = this.convToPoint3D (newPos, origin, obj, stepSize);

					// Check for collision
					if (collides [x, y] == -1) {
						collides [x, y] = this.checkCollision (newPt, blockId);
						if (collides [x, y] == 0) {
							continue;
						}
					}

					// Compute g score of the point
					double tentativeGScore = gScore [pos.getX (), pos.getY ()]
					                         + newPt.calc2DL2Distance (pt);

					// If newPt is in not in open set
					bool newEntry = false;
					if (!openSet [x, y]) {
						openSet [x, y] = true;
						newEntry = true;
					} else if (tentativeGScore >= gScore [x, y]) {
						continue;
					}

					// Admissible heuristic: compute l2 distance from the final  
					double h = newPt.calc2DL2Distance (objf);

					gScore [x, y] = (float)tentativeGScore;
					fScore [x, y] = (float)(tentativeGScore + h);
					parent [x, y] = pos;
					action [x, y] = blockId * 4 + actionIx;

					// Double check the line below. Add if not expanded.
					if (!newEntry) {
						priorityQueue.delete (pointer [x, y]);
					}

					Node n = new Node (newPt, newPos, fScore [x, y]);
					pointer [x, y] = n;
					priorityQueue.insert (n);
					trajSize [x, y] = 1 + trajSize [pos.getX (), pos.getY ()];
				}
			}

			List<int> trajectory = new List<int> ();
			List<Point3D> trajectoryPoints = new List<Point3D> ();

			if (goal == null) {
				trajectory.Add (80); //80 represents stop action
				trajectoryPoints.Add (obj);
				trajectoryPoints.Add (obj);
			} else {
				Pos it = goal;
				while (it != null) {
					Pos nIt = parent [it.getX (), it.getY ()];
					if (nIt != null) {

						int actionIx = action [it.getX (), it.getY ()];

						if (actionIx == -1) {
							throw new ApplicationException ("-1 action found. This is a bug");
						}
						trajectory.Add (actionIx);
						trajectoryPoints.Add (pointer [it.getX (), it.getY ()].getPoint ()); // add it
					}
					it = nIt;
				}
				trajectoryPoints.Add (obj); 

				trajectory.Reverse ();
				trajectory.Add (80); //80 represents stop action
				trajectoryPoints.Reverse ();
				Point3D lastPt = trajectoryPoints [trajectoryPoints.Count - 1];
				trajectoryPoints.Add (lastPt);
			}

			foreach (int i in trajectory) {
				if (i != 80) {
					totalTrajectory.Add (i);
				}
			}
		}
			
		return totalTrajectory.Count;
	}

	// Uses A* algorithm to find the optimal path.
	public TrajectoryResult findShortestPathAStar(ObjectSystem final) {

		if (this.disableTrajectory) {
			return new TrajectoryResult(this.dummyList,  null, Double.PositiveInfinity);
		}

		Debug.Log ("Finding shortest path using A*");

		// Find the index of the object that is different between the two environments
		// there should be exactly one
		int blockId = 0;
		for(int k = 0; k < this.objectLocations.Count; k++) {

			Point3D o1 = this.objectLocations [k];
			Point3D o2 = final.objectLocations [k];

			if (o1.calc2DL2Distance (o2) > 0.01f) {
				blockId = k;
				break;
			}
		}

		Point3D obj = this.objectLocations [blockId];
		Point3D objf = final.objectLocations [blockId];

		const int maxTrajSize = 40;//20;
		const double tol = 0.001;

		bool[,] closedSet = new bool[2 * maxTrajSize + 1, 2 * maxTrajSize + 1];
		bool[,] openSet = new bool[2 * maxTrajSize + 1, 2 * maxTrajSize + 1];
		float[,] gScore = new float[2 * maxTrajSize + 1, 2 * maxTrajSize + 1];
		float[,] fScore = new float[2 * maxTrajSize + 1, 2 * maxTrajSize + 1];
		int[,] action = new int[2 * maxTrajSize + 1, 2 * maxTrajSize + 1];
		Pos[,] parent = new Pos[2 * maxTrajSize + 1, 2 * maxTrajSize + 1];
		Node[,] pointer = new Node[2 * maxTrajSize + 1, 2 * maxTrajSize + 1];
		int[,] trajSize = new int[2 * maxTrajSize + 1, 2 * maxTrajSize + 1];
		//Collide: -1 not evaluate, 0 evaluated and not safe, 1 evaluated and safe 
		int[,] collides = new int[2 * maxTrajSize + 1, 2 * maxTrajSize + 1];

		for(int j = 0; j < 2 * maxTrajSize + 1; j++) {
			for (int k = 0; k < 2 * maxTrajSize + 1; k++) {
				closedSet [j, k] = false;
				openSet [j, k] = false;
				gScore [j, k] = float.PositiveInfinity;
				fScore [j, k] = float.PositiveInfinity;
				action [j, k] = -1;
				parent [j, k] = null;
				pointer [j, k] = null;
				trajSize [j, k] = -1;
				collides [j, k] = -1;
			}
		}

		MinHeap<Node> priorityQueue = new MinHeap<Node> ();

		// Initialize with the starting state
		openSet[maxTrajSize, maxTrajSize] = true;
		gScore[maxTrajSize, maxTrajSize] = 0;
		fScore [maxTrajSize, maxTrajSize] = (float)obj.calc2DL2Distance (objf);

		Pos origin = Pos.of(maxTrajSize, maxTrajSize);
		Node originNode = new Node (obj, origin, fScore [maxTrajSize, maxTrajSize]);

		pointer [maxTrajSize, maxTrajSize] = originNode;
		trajSize [maxTrajSize, maxTrajSize] = 0;
		priorityQueue.insert (originNode);

		// Output values
		double minAchieveDist = Double.PositiveInfinity;
		Pos goal = null;
		bool reached = false;

		while (priorityQueue.size() > 0) {

			// Get the lowest cost point
			Node lowestCostPt = priorityQueue.extractMin();
			Pos pos = lowestCostPt.getPos ();
			Point3D pt = lowestCostPt.getPoint ();

			// Check if this is the closest distance reached to goal
			double goalDist = pt.calc2DL2Distance(objf);
			if (goalDist < minAchieveDist) {
				goal = pos;
				minAchieveDist = goalDist;
			}

			// Check if it is close enough to the goal (with margin tol)
			if (goalDist < tol) {
				goal = pos;
				reached = true;
				minAchieveDist = goalDist;
				Debug.Log ("Reached the goal state. goal dist " + goalDist + " and tol " + tol);
				break;
			}

			if (trajSize [pos.getX (), pos.getY ()] >= maxTrajSize) {
				continue;
			}

			// Remove it from openset
			openSet[pos.getX(), pos.getY()] = false;

			// Add it to closedSet
			closedSet[pos.getX(), pos.getY()] = true;

			// Find the neighbors of this point and add to score
			// removing those which lead to collision
			for (int actionIx = 0; actionIx < 4; actionIx++) {

				// Apply the action to generate the new Point3D
				Pos newPos = this.applyAction (pos, actionIx);

				if (newPos == null) {
					continue;
				}

				int x = newPos.getX ();
				int y = newPos.getY ();

				// Check if newPt is in closed set or has been shown to collide
				// before. If yes then continue
				if(closedSet[x, y] || collides[x,y] == 0) {
					continue;
				}

				Point3D newPt = this.convToPoint3D (newPos, origin, obj, stepSize);

				// Check for collision
				if (collides [x, y] == -1) {
					collides [x, y] = this.checkCollision (newPt, blockId);
					if (collides [x, y] == 0) {
						continue;
					}
				}

				// Compute g score of the point
				double tentativeGScore = gScore[pos.getX(), pos.getY()]
										+ newPt.calc2DL2Distance(pt);
				
				// If newPt is in not in open set
				bool newEntry = false;
				if (!openSet [x, y]) {
					openSet [x, y] = true;
					newEntry = true;
				} else if (tentativeGScore >= gScore [x, y]) {
					continue;
				}

				// Admissible heuristic: compute l2 distance from the final  
				double h = newPt.calc2DL2Distance(objf);

				gScore [x, y] = (float)tentativeGScore;
				fScore [x, y] = (float)(tentativeGScore + h);
				parent [x, y] = pos;
				action [x, y] = blockId * 4 + actionIx;

				// Double check the line below. Add if not expanded.
				if (!newEntry) {
					priorityQueue.delete (pointer[x, y]);
				}

				Node n = new Node (newPt, newPos, fScore [x, y]);
				pointer [x, y] = n;
				priorityQueue.insert (n);
				trajSize [x, y] = 1 + trajSize [pos.getX (), pos.getY ()];
			}
		}

		if (!reached) {
			Debug.Log ("Did not reach the goal state. Min dist " + minAchieveDist);
		}

		List<int> trajectory = new List<int> ();
		List<Point3D> trajectoryPoints = new List<Point3D> ();

		if (goal == null) {
			trajectory.Add (80); //80 represents stop action
			trajectoryPoints.Add (obj);
			trajectoryPoints.Add (obj);
		} else {
			Pos it = goal;
			while(it != null) {
				Pos nIt = parent [it.getX (), it.getY ()];
				if (nIt != null) {
					
					int actionIx = action [it.getX (), it.getY ()];
				
					if (actionIx == -1) {
						throw new ApplicationException ("-1 action found. This is a bug");
					}
					trajectory.Add (actionIx);
					trajectoryPoints.Add (pointer [it.getX (), it.getY ()].getPoint ()); // add it
				}
				it = nIt;
			}
			trajectoryPoints.Add (obj); 
				
			trajectory.Reverse ();
			trajectory.Add (80); //80 represents stop action
			trajectoryPoints.Reverse ();
			Point3D lastPt = trajectoryPoints [trajectoryPoints.Count - 1];
			trajectoryPoints.Add (lastPt);
		}

		// Terminate trajectory if exceeding length 20
		if (trajectory.Count > 40) {
			trajectory = trajectory.GetRange (0, 40);
			trajectoryPoints = trajectoryPoints.GetRange (0, 41);
		}

		//this.checkTrajectoryForCollision (trajectory, collides);

		string trajS = "";
		for (int i = 0; i < trajectory.Count; i++) {
			trajS = trajS + " (" + trajectoryPoints [i].ToString () + ") ";
			trajS = trajS + " " + trajectory [i];
		}
		trajS = trajS + " " + trajectoryPoints [trajectoryPoints.Count - 1];
		Debug.Log ("Trajectory " + trajectory.Count + " and points " + trajectoryPoints.Count);
		Debug.Log ("Trajectory " + trajS);
			
		return new TrajectoryResult (trajectory, trajectoryPoints, minAchieveDist);
	}
		
	public List<int> findShortestPathNaiveGreedy(ObjectSystem final) {

		if (this.disableTrajectory) {
			return this.dummyList;
		}

		// TODO need to make this better
		// Find the shortest collision free path between current 
		// and final object system.

		// Find the index of the object that is different between the two environments
		// there should be exactly one
		int i = 0;
		for(int k = 0; k < this.objectLocations.Count; k++) {

			Point3D o1 = this.objectLocations [k];
			Point3D o2 = final.objectLocations [k];

			if (o1.calc2DL2Distance (o2) > 0.1f) {
				i = k;
				break;
			}
		}

		Point3D obj = this.objectLocations [i];
		Point3D objf = final.objectLocations [i];

		Point3D it = obj;

		double tol = 0.1f;

		List<int> trajectory = new List<int> ();

		while (trajectory.Count < 20) {

			int bestAction = -1;
			double bestActionDistance = Double.PositiveInfinity;
			Point3D bestPt = null;

			for (int action = 0; action < 4; action++) {

				//Apply the action to generate the new Point3D
				Point3D newPt = this.applyAction(it, action);

				// Compute the distance from the final  
				double newDistance = newPt.calc2DL2Distance(objf);

				if (newDistance < bestActionDistance) {
					
					// Check for collision
					bool isColliding = false;
					for (int j = 0; j < this.objectLocations.Count; j++) {
						if (j == i) {
							continue;
						}
						if(this.objectLocations[j].isColliding(newPt)) {
							isColliding = true;
							break;
						}
					}

					if (isColliding) {
						continue;
					}

					bestActionDistance = newDistance;
					bestAction = action;
					bestPt = newPt;
				}
			}

			if (bestAction == -1) {
				throw new ApplicationException ("Best Action is -1");
			}

			it = bestPt;

			//Since you entered this region, there is always a best Action.
			//In worst case, involves you going back a step.
			trajectory.Add(i * 4 + bestAction);

			Debug.Log ("Best Action Distance " + bestActionDistance);

			//If the new distance is below tol then add stop action which has id 80
			if(bestActionDistance < tol) {
				trajectory.Add(80);
				break;
			}
		}

		return trajectory;
	}


	// Given a startObj centered at center. Return the point3d location of the pos.
	// 1 unit of pos is given by stepSize in point3d.
	public Point3D convToPoint3D(Pos pos, Pos center, Point3D startObj, double stepSize) {
		return new Point3D (startObj.getX () + (pos.getX () - center.getX ()) * stepSize, 
							startObj.getY (),
							startObj.getZ () + (pos.getY () - center.getY ()) * stepSize);
	}

	private Pos convToPos(Point3D pt, Pos center, Point3D startObj, double stepSize) {
		int x = Convert.ToInt32 ((pt.getX () - startObj.getX ()) / stepSize + (double)center.getX ());
		int y = Convert.ToInt32 ((pt.getZ () - startObj.getZ ()) / stepSize + (double)center.getY ());
		return Pos.of (x, y);
	}

	public Pos applyAction(Pos pos, int action) {

		int x = pos.getX ();
		int y = pos.getY ();

		switch (action) {
		case 0:
			return Pos.of (x + 1, y);
		case 1:
			return Pos.of (x - 1, y);
		case 2:
			return Pos.of (x, y - 1);
		case 3:
			return Pos.of (x, y + 1);
		default:
			throw new ApplicationException ("Action has to be in {0, 1, 2, 3}");
		}
	}
}

public class TrajectoryResult {

	private List<int> trajectory;
	private List<Point3D> trajectoryPoints;
	private double minDistanceAchieved;

	public TrajectoryResult(List<int> trajectory, List<Point3D> trajectoryPoints, double minDistanceAchieved) {
		this.trajectory = trajectory;
		this.trajectoryPoints = trajectoryPoints;
		this.minDistanceAchieved = minDistanceAchieved;
	}

	public List<int> getTrajectory() {
		return this.trajectory;
	}

	public List<Point3D> getTrajectoryPoints() {
		return this.trajectoryPoints;
	}

	public double getMinDistanceAchieved() {
		return this.minDistanceAchieved;
	}
}

public class Pos {

	private int x;
	private int y;

	private Pos(int x, int y) {
		this.x = x;
		this.y = y;
	}

	public static Pos of(int x, int y) {
		return new Pos (x, y);
	}

	public int getX() {
		return this.x;
	}

	public int getY() {
		return this.y;
	}
}

public class Node : IComparable {

	private Point3D pt;
	private Pos pos;
	private double score;

	public Node(Point3D pt, Pos pos, double score) {
		this.pt = pt;
		this.pos = pos;
		this.score = score;
	}

	public Point3D getPoint() {
		return this.pt;
	}

	public Pos getPos() {
		return this.pos;
	}

	public double getScore() {
		return this.score;
	}

	public int CompareTo(object o) {

		Node n = null;
		try {
			n = (Node)o;
		} catch(Exception e) {
			throw new ApplicationException ("Error while casting. Error: " + e);
		}

		if (this.score == n.score) {
			return 0;
		}

		if (this.score < n.score) {
			return -1;
		} else {
			return 1;
		}
	}
}
