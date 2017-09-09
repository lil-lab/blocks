using UnityEngine;
using System.IO;
using System.Collections;
using System.Threading;
using System.Collections.Generic;

namespace RobotControllerFunctionality {

	class Job {

		private TranslationStepAction action;

		public Job(TranslationStepAction action) {
			this.action = action;
		}

		public TranslationStepAction getAction() {
			return this.action;
		}
	}

	public class RobotController : MonoBehaviour {
		// TODO robot controller is currently doing things
		// that a separate kernel class should do e.g., get reward etc.

		LinearTranslation translation = null;

		Queue<Job> queue = null;

		SynchronizedInt numberOfJobsFinished = null;

		MDPManager mdpManager = null;

		int snapshotID = 1;
		float blockSize = 1.0f;

		// Last Object System
		ObjectSystem current;

		// Start Object System
		ObjectSystem end = null;

		// Message header
		private MessageHeader messageHeader;

		// Maximum number of actions per episode
		private int maxNumActions;

		// Number of actions in this episode
		private int numActions;

		// List of cubes
		private List<GameObject> cubes;

		// Step size for translation
		private float stepSize;

		// Action ID
		private int actionID;

		// Reward string: Contains reward for all actions
		private string rewardStr;

		// Minimum approach bisk metric
		private double minimumApproachDistance;

		// Use this for initialization
		public void init (List<GameObject> cubes, float blockSize, float stepSize, int horizon) {

			//Add translation action controller
			GameObject robot = GameObject.Find ("Robot");
			if (robot == null) {
				Debug.LogError ("There must be a robotic agent called 'Robot'");
			}

			this.queue = new Queue<Job> ();
			this.numberOfJobsFinished = new SynchronizedInt ();

			// Add actions that the robot can perform e.g., translation, picking etc.
			this.translation = robot.AddComponent<LinearTranslation> ();
			this.translation.init (10.00f);

			// Snapshot ID
			this.snapshotID = 1;

			// Maximum number of epsiodes
			this.maxNumActions = horizon;
			this.numActions = 0;
			this.blockSize = blockSize;

			this.minimumApproachDistance = Mathf.Infinity;

			this.cubes = cubes;
			this.stepSize = stepSize;
			this.current = this.getCurrentObjectSystem ();
			this.rewardStr = "$NONE$";
		}

		private ObjectSystem getCurrentObjectSystem() {

			List<Point3D> cubeLocations = new List<Point3D> ();

			for(int i = 0; i < this.cubes.Count; i++) {

				GameObject cube = this.cubes [i];

				if (!cube.activeSelf)
					break;

				Vector3 pos = cube.transform.position;
				Point3D pt = new Point3D(pos.x, pos.y, pos.z);
				cubeLocations.Add(pt);
			}

			ObjectSystem currentSystem = new ObjectSystem (this.blockSize, cubeLocations);
			return currentSystem;
		}

		public void initMDPManager(double gamma, double failureReward, double winReward, double stopActionReward, 
						int rewardFunctionType, bool useSimplifiedProblem, int horizon) {
			
			// Add MDP manager that computes rewards
			this.mdpManager = new MDPManager(gamma, failureReward, winReward, stopActionReward, 
				rewardFunctionType, useSimplifiedProblem, horizon);
		}

		public int getNumberOfJobsFinished() {
			return this.numberOfJobsFinished.getVal ();
		}

		// When this function is called, no jobs should be running.
		public void resetNumberOfJobsFinished() {

			this.numberOfJobsFinished.reset ();

			//Reset jobs on all actions
			this.translation.resetNumberOfJobsFinished();
		}

		public void setNumberOfJobsFinished(int newVal) {
			this.numberOfJobsFinished.setVal (newVal);
		}

		public void addJob(string actionDescription) {
			Job job = new Job (TranslationStepAction.parse (actionDescription));
			this.actionID = job.getAction ().getActionID ();

			// Get reward string
			//this.rewardStr = this.calcRewardForAllActions ();

			lock (this.queue) {
				this.queue.Enqueue (job);
			}
			this.numActions++;
		}

		public void updateRewardString() {
//			this.rewardStr = this.calcRewardForAllActions ();
		}

		public double calcReward(bool stopAction) {
			ObjectSystem newState = this.getCurrentObjectSystem ();
			if (stopAction) {
				this.actionID = 80;
			}

			float reward = this.mdpManager.calcRewardManager (this.current, newState, this.actionID, 
				               this.messageHeader, stopAction, this.numActions);
			this.current = newState;

			// Updated closest approach metric
			this.updateClosestApproachMetric ();

			return reward;
		}

		// Computes reward for all actions. Used by contextual bandit algorithms for sanity check
		private string calcRewardForAllActions() {

			string rewardStr = "";

			int indexObjMoved = this.mdpManager.getIndexOfBlockMoved ();

			this.mdpManager.turnOffOverridePotential ();

			for (int i = 0; i <= 80; i++) {
				
				if (i == 80) {
					float reward = this.mdpManager.calcRewardManager (this.current, this.current, i, 
						               messageHeader, true, this.numActions);
					rewardStr = rewardStr + reward;
				} else {

					// Take the action
					int blockID = i / 4;
					int direction = i % 4;

					Vector3 directionVec = new Vector3 (0, 0, 0);

					switch (direction) {
					case 0:
						directionVec = new Vector3 (this.stepSize, 0.0f, 0.0f);
						break;
					case 1:
						directionVec = new Vector3 (-this.stepSize, 0.0f, 0.0f);
						break;
					case 2:
						directionVec = new Vector3 (0.0f, 0.0f, -this.stepSize);
						break;
					case 3:
						directionVec = new Vector3 (0.0f, 0.0f, this.stepSize);
						break;
					default:
						Debug.LogError ("Direction has to be one of {0, 1, 2, 3}");
						break;
					}

					Point3D directionPt = new Point3D (directionVec.x, directionVec.y, directionVec.z);

					// Do virtual execution to find if collision fails
					GameObject gameObj = this.cubes [blockID];

					MessageHeader messageHeader = this.translation.quickExec (gameObj, directionVec, blockID,
						                              this.current.getObjectLocations (), this.blockSize);

					// Compute 
					List<Point3D> curObjLoc = this.current.getObjectLocations ();
					List<Point3D> newObjLoc = new List<Point3D> ();
					int ix = 0;

					foreach (Point3D p in curObjLoc) {
						if (ix == blockID) {
							newObjLoc.Add (p.add (directionPt));
						} else {
							newObjLoc.Add (p);
						}
						ix++;
					}

					ObjectSystem newState = new ObjectSystem (this.current.getSizeOfBlock (), newObjLoc);

					float reward = this.mdpManager.calcRewardManager (this.current, newState, i, 
						               messageHeader, false, this.numActions);
					rewardStr = rewardStr + reward + "%";
				}
			}
				
			this.mdpManager.turnOnOverridePotential ();

			return rewardStr;
		}

		public string getRewardStr() {
			return this.rewardStr;
		}

		public double calcBiskMetric() {
			ObjectSystem newState = this.getCurrentObjectSystem ();
			double biskMetric = this.mdpManager.calcBiskMetric (newState);

			return biskMetric;
		}

		public void updateClosestApproachMetric() {
			ObjectSystem newState = this.getCurrentObjectSystem ();
			double biskMetric = this.mdpManager.calcBiskMetric (newState);

			this.minimumApproachDistance = (double)Mathf.Min ((float)this.minimumApproachDistance, (float)biskMetric);
		}

		public double calcClosestApproachMetric() {
			this.updateClosestApproachMetric ();

			return this.minimumApproachDistance;
		}

		public double calcActionErrorMetric() {
			ObjectSystem newState = this.getCurrentObjectSystem ();
			int numActionsToReachGoal = newState.findGeneralShortestPathAStar (this.end);

			return (double)numActionsToReachGoal;
		}

		public double calculateShortestPathMetric() {

			ObjectSystem newState = this.getCurrentObjectSystem ();

			//Compute distance between this newState and the old state
			TrajectoryResult result = newState.findShortestPathAStar (this.end);
			List<int> trajectory = result.getTrajectory ();
			int count = trajectory.Count;
			if (trajectory [trajectory.Count - 1] == 80) {
				count--;
			}
			double distance = this.stepSize * count + result.getMinDistanceAchieved ();
			double shortestPathDistance = distance / this.blockSize;
			
			return shortestPathDistance;
		}

		public void changeEpisode(List<GameObject> cubes, ObjectSystem start, ObjectSystem end,
								  List<int> trajectory, List<Point3D> trajectoryPoints, int numObjects, float blockSize) {
			this.cubes = cubes;
			this.mdpManager.changeScenario (start, end, trajectory, trajectoryPoints);
			this.blockSize = blockSize;
			this.end = end;
			this.minimumApproachDistance = this.mdpManager.calcBiskMetric (start);
		}

		public MessageHeader getMessageHeader() {
			return this.messageHeader;
		}

		public int getActionCounter() {
			return this.numActions;
		}

		public void resetActionCounter() {
			this.numActions = 0;
		}

		// Checks if reset need to be done
		public bool shouldReset() {
			
			if (this.numActions % this.maxNumActions == 0) {
				return true;
			} else {
				return false;
			}
		}

		public string reset() {
			InitScript init = GameObject.Find ("Ground").GetComponent<InitScript> ();
			string instruction = init.reset ();

			string fileName = "deprecated";
			this.snapshotID++;

			this.current = this.getCurrentObjectSystem ();
			return fileName + "#" + instruction;
		}

		public void resetStart() {
			InitScript init = GameObject.Find ("Ground").GetComponent<InitScript> ();
			init.resetStart ();
		}

		public void resetEnd() {
			InitScript init = GameObject.Find ("Ground").GetComponent<InitScript> ();
			init.resetEnd ();
		}

		public void changeRewardFunctionType(int newRewardFunctionType) {
			this.mdpManager.changeRewardFunction (newRewardFunctionType);
		}
		
		// Update is called once per frame
		void Update () {

			//Dispatch all jobs
			lock (this.queue) {
				while (this.queue.Count > 0) {
					Job job = this.queue.Dequeue ();
					this.executeAction (job.getAction());
				}
			}

			//Check how many jobs were finished
			int translationFinished = this.translation.getNumberOfJobsFinished();
			this.setNumberOfJobsFinished (translationFinished);
			if (translationFinished == 1) {
				this.messageHeader = this.translation.getMessageHeader ();
			}
		}

		private void executeAction(TranslationStepAction action) {

			GameObject cube = this.cubes [action.getCubeID ()];
			TranslationStepAction.Direction direction = action.getDirection ();

			float xNew = 0.0f, zNew = 0.0f;

			switch(direction) {
			case TranslationStepAction.Direction.North:
				xNew = this.stepSize;
				break;
			case TranslationStepAction.Direction.South:
				xNew = - this.stepSize;
				break;
			case TranslationStepAction.Direction.East:
				zNew = - this.stepSize;
				break;
			case TranslationStepAction.Direction.West:
				zNew = this.stepSize;
				break;
			default: 
				Debug.LogError ("Unknown direction " + direction);
				break;
			}

			Vector3 vec = new Vector3 (xNew, 0.0f, zNew);
			this.translation.addJob(cube, vec);
		}
	}
}
