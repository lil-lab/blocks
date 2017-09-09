using System;
using System.Collections.Generic;
using UnityEngine;

/** Class for taking care of Markov Decision Process. Provides
 * functions for computing rewards */
public class MDPManager {

	// Discounting factor in MDP
	private double gamma;

	// Starting system 
	private ObjectSystem start;

	// Final system which we are targetting
	private ObjectSystem end;

	// Reward for failure cases such as collision
	private double failureReward;

	// Reward for winning
	private double winReward;

	// Stop action reward
	private float stopActionReward;

	// Object to move index
	private  int indexObjMoved;

	// Reward function type 
	private int rewardFunctionType;

	// Use simplified problem
	private bool useSimplifiedProblem;

	// Time horizon
	private int horizon;

	// Trajectory of Demonstration/Oracle
	private List<int> trajectory;
	private List<Point3D> trajectoryPoints;

	public bool overridePotential;

	// Potential
	private double potentialOld;

	public MDPManager (double gamma, double failureReward, double winReward, double stopActionReward, 
						int rewardFunctionType, bool useSimplifiedProblem, int horizon) {
		this.gamma = gamma;
		this.failureReward = failureReward;
		this.winReward = winReward;
		this.stopActionReward = (float)stopActionReward;
		this.rewardFunctionType = rewardFunctionType;
		this.useSimplifiedProblem = useSimplifiedProblem;
		this.horizon = horizon;
		this.overridePotential = true;
		Debug.Log ("MDP Manager created with reward type " + rewardFunctionType);
	}

	public void changeRewardFunction(int newRewardFunctionType) {
		int oldRewardFunctionType = this.rewardFunctionType;
		this.rewardFunctionType = newRewardFunctionType;
		Debug.Log ("MDP Manager. Changed the reward function type from " + oldRewardFunctionType + " to " + this.rewardFunctionType);
	}

	public void changeScenario(ObjectSystem start, ObjectSystem end, List<int> trajectory, List<Point3D> trajectoryPoints) {
		this.start = start;
		this.end = end;
		this.potentialOld = -0.02f;//0.0;
		this.trajectory = trajectory;
		this.trajectoryPoints = trajectoryPoints;

		// Compute the objected moved
		List<Point3D> objLoc1 = this.start.getObjectLocations ();
		List<Point3D> objLoc2 = this.end.getObjectLocations ();

		double blockSize1 = this.start.getSizeOfBlock ();
		double blockSize2 = this.end.getSizeOfBlock ();

		if (objLoc1.Count != objLoc2.Count) {
			throw new ApplicationException ("MDP Manager expects same number of objects." +
				" Either environments are different" + 
				" or an object fell off the board and was removed");
		}

		if (Math.Abs (blockSize1 - blockSize2) > 0.001) {
			throw new ApplicationException ("Block size differ. Found " + blockSize1 + " and " + blockSize2);
		}

		int mainBlock = -1;
		int changedBlock = 0;

		for (int i = 0; i < objLoc1.Count; i++) {

			Point3D p1 = objLoc1 [i];
			Point3D p2 = objLoc2 [i];

			if (p1.calc2DL2Distance (p2) > 0.001) {
				mainBlock = i;
				changedBlock++;

				if (changedBlock > 1) {
					mainBlock = -1;
					break;
				}
			}
		}

		if (mainBlock == -1) {
			Debug.LogError ("Needed exactly one block to move. Found "
				+ changedBlock + " many blocks moved.");
		}

		this.indexObjMoved = mainBlock;
	}

	public double getDiscountingFactor() {
		return this.gamma;
	}

	public void turnOnOverridePotential() {
		this.overridePotential = true;
	}

	public void turnOffOverridePotential() {
		this.overridePotential = false;
	}

	public int getIndexOfBlockMoved() {
		return this.indexObjMoved;
	}

	// Calculuates the distance between two environment which is 
	// given by average distance between corresponding objects. For this, 
	// the environments must have same the same set of objects with unique objects.
	public double calcDist(ObjectSystem current, ObjectSystem toReach) {

		List<Point3D> objLoc1 = current.getObjectLocations ();
		List<Point3D> objLoc2 = toReach.getObjectLocations ();

		if (objLoc1.Count != objLoc2.Count) {
			throw new ApplicationException ("MDP Manager expects same number of objects." +
				" Either environments are different" + 
				" or an object fell off the board and was removed");
		}

		double distance = 0;
		for (int i = 0; i < objLoc1.Count; i++) {

			Point3D obj1 = objLoc1 [i];
			Point3D obj2 = objLoc2 [i];

			distance = distance + obj1.calcL2Distance (obj2);
		}

		return distance / (double)Math.Max(objLoc1.Count, 1);
	}

	// Calculate Bisk metric that is distance between the main block between final state
	// and to reach. Divide this by block size.
	public double calcBiskMetric(ObjectSystem finalState) {

		if (!this.useSimplifiedProblem) {
			double sumOfEuclidean = this.calcUnnormalizedSumOfEuclidean (finalState, this.end);
			double blockSize_ = finalState.getSizeOfBlock ();
			double normalizedMetric = sumOfEuclidean / blockSize_;
			return normalizedMetric;
		}

		List<Point3D> objLoc2 = this.end.getObjectLocations ();
		double blockSize = this.end.getSizeOfBlock ();

		List<Point3D> finalStateObjLoc = finalState.getObjectLocations ();
		Point3D finalPos = finalStateObjLoc [this.indexObjMoved];
		Point3D mainBlockAfter = objLoc2 [this.indexObjMoved];

		double dist = finalPos.calc2DL2Distance (mainBlockAfter);
		double biskMetric = dist / blockSize;

		return biskMetric;
	}

	// Calculate difference between two block states by squared
	public double calcUnnormalizedSumOfEuclidean(ObjectSystem state1, ObjectSystem state2) {

		List<Point3D> objLoc1 = state1.getObjectLocations ();
		List<Point3D> objLoc2 = state2.getObjectLocations ();

		if (objLoc1.Count != objLoc2.Count) {
			throw new ApplicationException ("MDP Manager expects same number of objects." +
			" Either environments are different" +
			" or an object fell off the board and was removed. Found " + objLoc1.Count + " and " + objLoc2.Count);
		}

		double blockSize1 = state1.getSizeOfBlock ();
		double blockSize2 = state2.getSizeOfBlock ();

		if (Math.Abs (blockSize1 - blockSize2) > 0.001) {
			throw new ApplicationException ("Block size differ. Found " + blockSize1 + " and " + blockSize2);
		}

		double sumOfEuclidean = 0;

		for (int i = 0; i < objLoc1.Count; i++) {

			Point3D p1 = objLoc1 [i];
			Point3D p2 = objLoc2 [i];

			double dist = p1.calc2DL2Distance (p2);
			sumOfEuclidean = sumOfEuclidean + dist;
		}
			
		return sumOfEuclidean;
	}

	// Calculates reward for a given action.
	private float calcCosineDistanceReward1(ObjectSystem current, ObjectSystem newState, MessageHeader header, bool stopAction) {

		ObjectSystem toReach = this.end;

		// The system went from current to newState and it has to reach toReach

		List<Point3D> objLoc1 = current.getObjectLocations ();
		List<Point3D> objLoc2 = newState.getObjectLocations ();
		List<Point3D> objLoc3 = toReach.getObjectLocations ();

		if (objLoc1.Count != objLoc2.Count || objLoc1.Count != objLoc3.Count) {
			Debug.Log ("Found " + objLoc1.Count + " but there was  " + objLoc2.Count + " and " + objLoc3.Count);
			throw new ApplicationException ("MDP Manager expects same number of objects." +
				" Either environments are different" + 
				"or an object fell off the board and was removed");
		}

		if (header == MessageHeader.FailureACK) {
			return (float)this.failureReward;
		}

		double count = 0.0d;
		double tol = 0.0001d;
		double distance = 0.0d;
		int diffObject = 0;

		double maxDistance = 0.0d;

		// We compute reward using \sum_i Delta(o^1_i, o^2_i, o^3_i) where
		// i iterates over objects which moved between env1 and env2. 
		// So if we move one object from one place to another place where it should be
		// then we get a reward of 1.0. If we move one object away then we get a reward of -1.
		for (int i = 0; i < objLoc1.Count; i++) {

			Point3D obj1 = objLoc1 [i];
			Point3D obj2 = objLoc2 [i];
			Point3D obj3 = objLoc3 [i];

			double d12 = obj1.calc2DL2Distance (obj2);

			// if obj1 is away from obj3 
			//			and obj2 became closer to obj3 than obj1 then count 1
			//			and obj2 became away from obj3 than obj1 then count -1
			//			and obj2 is as far from obj3 as obj1 then count -0.5
			// if obj1 is same as obj3
			//			and obj2 became away from obj3 than obj1 then count -1
			//			and all are at same position then count 0

			double d23 = obj2.calc2DL2Distance (obj3);
			double d31 = obj3.calc2DL2Distance (obj1);

			maxDistance = Math.Max (maxDistance, d23);

			distance = distance + d23;
			if (d12 < tol) {
				continue;
			}

			diffObject++;

			if(d31 > tol) { //obj1 is away from obj3

				// We give reward based on cosine distance
				Point3D delta13 = obj3.sub (obj1);
				Point3D delta12 = obj2.sub (obj1);
				double cosineDistance = delta13.cosine2D (delta12);
				count = count + cosineDistance;
			} else if(d31 < tol) {
				if (d23 > d31 + tol) {
					count = count - 1.0;
				}
			}
		}

		distance = distance / (double)Math.Max(objLoc1.Count, 1);

		// Stopping when distance is less than tolerance is our winning condition
		// else stopping results in failure penalty.
		if (stopAction) {
			Debug.Log ("Win reward distance " + distance + " max distance " + maxDistance);
			if (maxDistance < 0.085) {		//INCREASE THIS TOL
				return (float)this.winReward;
			} else {
				return -1.0f;//(float)this.failureReward;
			}
		}

		if (diffObject == 0) { //TODO Basically no change occured. This is strange.
			return (float)this.failureReward;
		} else {
			float reward = ((float)count)/((float)diffObject);
			return reward;
		}
	}

	// This reward is based on distance potential-based shaping term.
	private float calcPotentialBasedRewardReduceDistance(ObjectSystem current, ObjectSystem newState,
		MessageHeader header, bool stopAction, int numActions) {

		if (header == MessageHeader.FailureACK || header == MessageHeader.FailureAndResetACK) {
			return -1.0f;
		}

		ObjectSystem toReach = this.end;

		// The system went from current to newState and it has to reach toReach

		// Compute distance d13 is pre-transition distance and d23 is post transition action. 
		double d13 = this.calcUnnormalizedSumOfEuclidean (current, toReach);
		double d23 = this.calcUnnormalizedSumOfEuclidean (newState, toReach);

		double blockSize = newState.getSizeOfBlock ();

		double rawReward = -0.02f;

		if (stopAction/* || numActions == this.horizon*/) {
			if (d23 < blockSize) { //1.0 if within 1 block distance
				rawReward = 1.0f;
			} else {
				rawReward = this.stopActionReward;
			}
		} else {
			rawReward = -0.02f; //verbosity penalty
		}

		double potentialOld = -d13 / blockSize;
		double potentialNew = -d23 / blockSize;

		//when stopping the new and old potential will be same.
		double shapedReward = rawReward + potentialNew - potentialOld; 
		return (float)shapedReward;
	}

	// This reward is based on potential. This reward is given based on 
	// how much do you reduce the distance to the goal.
	private float calcPotentialBasedRewardReduceDistanceV2(ObjectSystem current, ObjectSystem newState,
		MessageHeader header, bool stopAction, int numActions) {

		ObjectSystem toReach = this.end;

		// The system went from current to newState and it has to reach toReach
		List<Point3D> objLoc1 = current.getObjectLocations ();
		List<Point3D> objLoc2 = newState.getObjectLocations ();
		List<Point3D> objLoc3 = toReach.getObjectLocations ();

		// Compute distance d13 is pre-transition distance and d23 is post transition action. 
		double d13 = objLoc1 [this.indexObjMoved].calc2DL2Distance (objLoc3 [this.indexObjMoved]);
		double d23 = objLoc2 [this.indexObjMoved].calc2DL2Distance (objLoc3 [this.indexObjMoved]);
		double blockSize = newState.getSizeOfBlock ();

		double potentialOld = -d13 / blockSize;
		double potentialNew = -d23 / blockSize;

		//when stopping the new and old potential will be same.
		double reshapedReward = potentialNew - potentialOld; 
		return (float)reshapedReward;
	}

	// This reward is negative of distance from the new state to the goal state
	private float calcNegDistanceReward(ObjectSystem current, ObjectSystem newState,
		MessageHeader header, bool stopAction, int numActions) {

		// Compute distance d13 is pre-transition distance and d23 is post transition action. 
		double distance = this.calcUnnormalizedSumOfEuclidean (newState, this.end);
		double blockSize = newState.getSizeOfBlock ();
		double reward = -distance / blockSize;

		return (float)reward;
	}

	private float calcPotentialBasedRewardFollowOracleReduceDistanceV2(ObjectSystem current, 
		ObjectSystem newState, int action, MessageHeader header, bool stopAction, int numActions) {

		ObjectSystem toReach = this.end;

		// The system went from current to newState and it has to reach toReach
		List<Point3D> objLoc1 = current.getObjectLocations ();
		List<Point3D> objLoc2 = newState.getObjectLocations ();
		List<Point3D> objLoc3 = toReach.getObjectLocations ();

		// Compute distance d13 is pre-transition distance and d23 is post transition action. 
		double d13 = objLoc1 [this.indexObjMoved].calc2DL2Distance (objLoc3 [this.indexObjMoved]);
		double d23 = objLoc2 [this.indexObjMoved].calc2DL2Distance (objLoc3 [this.indexObjMoved]);
		double blockSize = newState.getSizeOfBlock ();

		double rawReward = -0.02f;

		if (stopAction /*|| numActions == this.horizon*/) {
			if (d23 < blockSize) { //1.0 if within 1 block distance
				rawReward = 1.0f;
			} else {
				rawReward = this.stopActionReward;
			}
		} else {
			rawReward = -0.02f; //verbosity penalty
		}


		double potentialOld = -d13 / blockSize;
		double potentialNew = -d23 / blockSize;

		//when stopping the new and old potential will be same.
		double reshapedReward = rawReward + (potentialNew - potentialOld)/(10.0);
		Point3D preTransitionState = objLoc1 [this.indexObjMoved];

		double oraclePotential = this.potentialFromTrajectoryV2 (preTransitionState,
			                         action, this.trajectoryPoints, this.trajectory);
		reshapedReward = reshapedReward + oraclePotential;

		return (float)reshapedReward;
	}

	// Given a trajectory and the point in which action was taken, finds the closest point 
	// on the trajectory if one exists and returns 1.0 if following the action
	// on the trajectory for that point else returns verbosity penalty of -0.02
	// TrajectoryPoints has a size one more than trajectory. 
	private double potentialFromTrajectoryV2(Point3D pt, int action, 
		List<Point3D> trajectoryPoints, List<int> trajectory) {

		int ix = 0;
		double minDist = Double.PositiveInfinity;
		int ctr = 0;
		foreach(Point3D p in trajectoryPoints) {
			double dist = pt.calc2DL2Distance (p);
			if (dist < minDist) { 
				// strict inequality means that during stop, it will not update
				// to the last but instead the second last point
				ix = ctr;
				minDist = dist;
			}
			ctr++;
		}

		double blockSize = this.start.getSizeOfBlock ();

		if (minDist < blockSize) { // The point is on the trajectory.
			// Check if the action taken is same as what is in the trajectory
			// If the point is the last point then return failure
			if (ix == trajectory.Count) {
				return -0.02f;
			}

			int actionIx = trajectory[ix];
			if (action == actionIx) { // Action taken matches the action taken 
				Debug.Log("Follows oracle V2 " + action + " win 0.5f");
				return 0.5f; // followed the oracle so win reward of 0.1f
			} else {
				Debug.Log ("Does not follow oracle V2 " + action);
				return -0.02f; // did not follow the oracle so fail reward of -0.02f
			}
		} else { //Not on trajectory so return verbosity penalty
			return -0.02f;
		}
	}

	private float calcPotentialBasedRewardFollowOracleReduceDistance(ObjectSystem current, ObjectSystem newState, int action, 
		MessageHeader header, bool stopAction, int numActions) {

		if (header == MessageHeader.FailureACK || header == MessageHeader.FailureAndResetACK) {
			return -1.0f;
		}

		ObjectSystem toReach = this.end;

		// The system went from current to newState and it has to reach toReach
		List<Point3D> objLoc1 = current.getObjectLocations ();

		// Compute distance d13 is pre-transition distance and d23 is post transition action. 
		double d13 = this.calcUnnormalizedSumOfEuclidean (current, toReach);
		double d23 = this.calcUnnormalizedSumOfEuclidean (newState, toReach);

		double blockSize = newState.getSizeOfBlock ();

		double rawReward = -0.02f;

		if (stopAction) {// || numActions == this.horizon) {
			if (d23 < blockSize) { //1.0 if within 1 block distance
				rawReward = 1.0f;
			} else {
				rawReward = this.stopActionReward;
			}
		} else {
			rawReward = -0.02f; //verbosity penalty
		}

		double potentialOld = -d13 / blockSize;
		double potentialNew = -d23 / blockSize;

		//when stopping the new and old potential will be same.
		double reshapedReward = rawReward + potentialNew - potentialOld; 
		Point3D preTransitionState = objLoc1 [this.indexObjMoved];

		double oraclePotential = this.potentialFromTrajectory (preTransitionState,
			action, this.trajectoryPoints, this.trajectory);
		reshapedReward = reshapedReward + oraclePotential - this.potentialOld;

		if (this.overridePotential) {
			// Hack that override potential. Hack was added to return rewards for all CB
			this.potentialOld = oraclePotential;
		}

		return (float)reshapedReward;
	}

	// This reward is based on potential. This reward is given based on 
	// following oracle.
	private float calcPotentialBasedRewardFollowOracle(ObjectSystem current, ObjectSystem newState, int action, 
		MessageHeader header, bool stopAction, int numActions) {

		if (header == MessageHeader.FailureACK || header == MessageHeader.FailureAndResetACK) {
			return -1.0f;
		}

		ObjectSystem toReach = this.end;

		// The system went from current to newState and it has to reach toReach
		List<Point3D> objLoc1 = current.getObjectLocations ();

		// Compute distance d13 is pre-transition distance and d23 is post transition action. 
		// double d23 = objLoc2 [this.indexObjMoved].calc2DL2Distance (objLoc3 [this.indexObjMoved]);
		double d23 = this.calcUnnormalizedSumOfEuclidean (newState, toReach);

		double blockSize = newState.getSizeOfBlock ();

		double rawReward = -0.02f;

		if (stopAction /*|| numActions == this.horizon*/) {
			if (d23 < blockSize) { //1.0 if within 1 block distance
				rawReward = 1.0f;
			} else {
				rawReward = this.stopActionReward;
			}
		} else {
			rawReward = -0.02f; //verbosity penalty
		}

		Point3D preTransitionState = objLoc1 [this.indexObjMoved];

		double potentialNew = this.potentialFromTrajectory (preTransitionState, action, this.trajectoryPoints, this.trajectory);
		double reshapedReward = rawReward + potentialNew - this.potentialOld;
		this.potentialOld = potentialNew;

		return (float)reshapedReward;
	}

	// Given a trajectory and the point in which action was taken, finds the closest point 
	// on the trajectory if one exists and returns 1.0 if following the action
	// on the trajectory for that point else returns verbosity penalty of -0.02
	// TrajectoryPoints has a size one more than trajectory. 
	private double potentialFromTrajectory(Point3D pt, int action, 
							List<Point3D> trajectoryPoints, List<int> trajectory) {

		int ix = 0;
		double minDist = Double.PositiveInfinity;
		int ctr = 0;
		foreach(Point3D p in trajectoryPoints) {
			double dist = pt.calc2DL2Distance (p);
			if (dist < minDist) { 
				// strict inequality means that during stop, it will not update
				// to the last but instead the second last point
				ix = ctr;
				minDist = dist;
			}
			ctr++;
		}
			
		double blockSize = this.start.getSizeOfBlock ();

		if (minDist < blockSize) { // The point is on the trajectory.
			// Check if the action taken is same as what is in the trajectory
			// If the point is the last point then return failure
			if (ix == trajectory.Count) {
				return -0.02f;
			}

			int actionIx = trajectory[ix];
			if (action == actionIx) { // Action taken matches the action taken 
				Debug.Log("Follows oracle " + action);
				return 1.0f; // followed the oracle so win reward of 1.0f
			} else {
				return -0.02f; // did not follow the oracle so fail reward of -0.02f
			}
		} else { //Not on trajectory so return verbosity penalty
			return -0.02f;
		}
	}

	// Calculates reward for a given action.
	// This reward computes cosine distance but reduces the magnitude so that the win still counts more.
	private float calcCosineDistanceReward(ObjectSystem current, ObjectSystem newState, MessageHeader header, bool stopAction) {

		ObjectSystem toReach = this.end;

		// The system went from current to newState and it has to reach toReach

		List<Point3D> objLoc1 = current.getObjectLocations ();
		List<Point3D> objLoc2 = newState.getObjectLocations ();
		List<Point3D> objLoc3 = toReach.getObjectLocations ();

		if (objLoc1.Count != objLoc2.Count || objLoc1.Count != objLoc3.Count) {
			Debug.Log ("Found " + objLoc1.Count + " but there was  " + objLoc2.Count + " and " + objLoc3.Count);
			throw new ApplicationException ("MDP Manager expects same number of objects." +
				" Either environments are different" + 
				"or an object fell off the board and was removed");
		}

		if (header == MessageHeader.FailureACK) {
			return -0.02f;
		}

		double count = 0.0d;
		double tol = 0.0001d;
		double distance = 0.0d;
		int diffObject = 0;

		double maxDistance = 0.0d;

		// We compute reward using \sum_i Delta(o^1_i, o^2_i, o^3_i) where
		// i iterates over objects which moved between env1 and env2. 
		// So if we move one object from one place to another place where it should be
		// then we get a reward of 1.0. If we move one object away then we get a reward of -1.
		for (int i = 0; i < objLoc1.Count; i++) {

			Point3D obj1 = objLoc1 [i];
			Point3D obj2 = objLoc2 [i];
			Point3D obj3 = objLoc3 [i];

			double d12 = obj1.calc2DL2Distance (obj2);

			// if obj1 is away from obj3 
			//			and obj2 became closer to obj3 than obj1 then count 1
			//			and obj2 became away from obj3 than obj1 then count -1
			//			and obj2 is as far from obj3 as obj1 then count -0.5
			// if obj1 is same as obj3
			//			and obj2 became away from obj3 than obj1 then count -1
			//			and all are at same position then count 0

			double d23 = obj2.calc2DL2Distance (obj3);
			double d31 = obj3.calc2DL2Distance (obj1);

			maxDistance = Math.Max (maxDistance, d23);

			distance = distance + d23;
			if (d12 < tol) {
				continue;
			}

			diffObject++;

			if(d31 > tol) { //obj1 is away from obj3

				// We give reward based on cosine distance
				Point3D delta13 = obj3.sub (obj1);
				Point3D delta12 = obj2.sub (obj1);
				double cosineDistance = delta13.cosine2D (delta12);
				count = count + cosineDistance;

			} else if(d31 < tol) {
				if (d23 > d31 + tol) {
					count = count - 1.0;
				}
			}
		}

		distance = distance / (double)Math.Max(objLoc1.Count, 1);

		// Stopping when distance is less than tolerance is our winning condition
		// else stopping results in failure penalty.
		if (stopAction) {
			Debug.Log ("Win reward distance " + distance + " max distance " + maxDistance);
			if (maxDistance < 0.085) {		//INCREASE THIS TOL
				return 1.0f;
			} else {
				return -1.0f;
			}
		}

		if (diffObject == 0) { //TODO Basically no change occured. This is strange.
			return -0.02f;
		} else {
			float reward = ((float)count)/((float)diffObject);
			reward = reward / ((float)this.horizon);
			return reward;
		}
	}

	// Calculates reward for a given action.
	private float calcSparseSignalReward(ObjectSystem current, ObjectSystem newState,
		MessageHeader header, bool stopAction, int numActions) {

		ObjectSystem toReach = this.end;

		// The system went from current to newState and it has to reach toReach
		List<Point3D> objLoc1 = current.getObjectLocations ();
		List<Point3D> objLoc2 = newState.getObjectLocations ();
		List<Point3D> objLoc3 = toReach.getObjectLocations ();

		if (objLoc1.Count != objLoc2.Count || objLoc1.Count != objLoc3.Count) {
			Debug.Log ("Found " + objLoc1.Count + " but there was  " + objLoc2.Count + " and " + objLoc3.Count);
			throw new ApplicationException ("MDP Manager expects same number of objects." +
			" Either environments are different" +
			"or an object fell off the board and was removed");
		}

		// Stopping when distance is less than tolerance is our winning condition
		// else stopping results in failure penalty.
		if (stopAction/* || numActions == this.horizon*/) {
			// Compute distance 

			double d23 = this.calcUnnormalizedSumOfEuclidean (newState, toReach);

			double winDistance = newState.getSizeOfBlock ();

			if (d23 < winDistance) {
				return 1.0f;
			} else {
				return this.stopActionReward;
			}
		}

		return -0.02f;
	}

	// This function computes different reward function based on chosen value.
	public float calcRewardManager(ObjectSystem current, ObjectSystem newState, int actionTaken, 
									MessageHeader header, bool stopAction, int numActions) {

		switch (this.rewardFunctionType) {
			case 0:	// Cosine distance
				return this.calcCosineDistanceReward(current, newState, header, stopAction);
			case 1: // Sparse signal
				return this.calcSparseSignalReward(current, newState, header, stopAction, numActions);
			case 2: // Sparse signal with distance potential
				return this.calcPotentialBasedRewardReduceDistance (current, newState, header, stopAction, numActions);
			case 3: // Sparse signal with oracle potential
				return this.calcPotentialBasedRewardFollowOracle(current, newState, actionTaken, header, stopAction, numActions);
			case 4: // Sparse signal with oracle potential
				return this.calcPotentialBasedRewardFollowOracleReduceDistance(current, newState,
												actionTaken, header, stopAction, numActions);
			case 5: // Sparse signal with oracle addition
				return this.calcPotentialBasedRewardFollowOracleReduceDistanceV2(current, newState,
					actionTaken, header, stopAction, numActions);
			case 6: // Sparse signal with oracle addition
				return this.calcPotentialBasedRewardReduceDistanceV2 (current, newState, header, stopAction, numActions);
			case 7: // Reward is negative of distance to the goal
				return this.calcNegDistanceReward(current, newState, header, stopAction, numActions);
			default:
				throw new ApplicationException ("Unknown reward type");
		}
	}
}
