using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.Threading;
using Newtonsoft.Json.Linq;
using RobotControllerFunctionality;

// Type of decoration to consider
public enum Decoration {Logos, Digits, Both, Blank};

public class InitScript : MonoBehaviour {

	private static List<string> brandsList = new List<string> (new string[] {"adidas", "bmw", "burgerking", "cocacola", 
		"esso", "heineken", "hp", "mcdonalds", "mercedesbenz", "nvidia", "pepsi", "shell", "sri", 
		"starbucks", "stella", "target", "texaco", "toyota", "twitter", "ups"});

	private AgentMessenger agentMessenger;
	private List<DataPoint> dataset;
	private int datapointIndex = 0;
	private bool first = true;

	private bool shuffleBeforeSelect = true;
	private int numDatapoints = 0;
	private string dataFileName = null;
	private float stepSize = 0.1f;

	// In simplified problem, you can only move one block.
	private bool simplifiedProblem = false;

	// List of all cubes (visible and invisible)
	private List<GameObject> logoCubes;
	private List<GameObject> digitCubes;

	// List of color of cubes
	private List<Color> colors;

	// Cached trajectories
	private CachedTrajectory cachedTraj;

	// Random number
	private static System.Random rnd;

	// Horizon
	private int horizon;

	// Use old reward function
	private int rewardFunctionType;

	// Use localhost
	private bool uselocalhost;

	// Stop reward
	private double stopActionReward;

	// row and col size
	private int screenSize;

	// Decoration used by the block
	private Decoration decoration;

	// List of digits text
	private List<GameObject> digitText;

	// List of textures of logos
	private List<Texture> logoTextures;

	// Repeat datapoint twice for self-critical training
	private bool repeat;
	private bool shownAlready = false;

	// Percentage of demonstrations to use. In case we want to use
	// only a few demonstrations.
	private float percentDemonstrations = 0;
	private int numDatapointsWithDemonstrations = 0;

	private List<Color> getUniqueColors() {
		List<Color> colors = new List<Color> {Color.blue, Color.green, Color.cyan, 
			Color.grey, Color.magenta, Color.black, Color.red, Color.yellow
		};

		// Brown 139 87	66
		colors.Add (new Color32 (139, 87, 66, 255));
		// Beet 142	56	142
		colors.Add (new Color32 (142, 56, 142, 255));
		// Sienna-2 238	121	66
		colors.Add (new Color32 (238, 121, 66, 255));
		// Banana 227 207 87	 
		colors.Add (new Color32 (227, 207, 87, 255));
		// Burlywood 222 184 135	
		colors.Add (new Color32 (222, 184, 135, 255));
		// Sgi Olivedrab 142 142 56	
		colors.Add (new Color32 (142, 142, 56, 255));
		// Dark-olive-green-3 62 205 90
		colors.Add (new Color32 (62, 205, 90, 255));
		// Torquise-blue 0	199	140
		colors.Add (new Color32 (0, 199, 140, 255));
		// Deep sky blue 0	154	205
		colors.Add (new Color32 (0, 154, 205, 255));
		// Cadet-blue-4 83	134	139
		colors.Add (new Color32 (83, 134, 139, 255));
		// Goldenrod-3 205	155	29
		colors.Add (new Color32 (205, 155, 29, 255));
		// Sgi Salmon 198	113	113
		colors.Add (new Color32 (198, 113, 113, 255));

		return colors;
	}

	private List<GameObject> createLogoCubes(ObjectSystem system, GameObject sampleCube) {

		this.colors = this.getUniqueColors ();
		List<GameObject> cubes = new List<GameObject> ();

		// Make the cube inactive
		sampleCube.SetActive (false);

		int ix = 0;
		float blockSize = (float)system.getSizeOfBlock ();

		foreach (string brand in brandsList) {
			
			Debug.Log ("Creating cube for brand: " + brand);
			Texture texture = (Texture2D)Resources.Load (brand);
			if (texture == null) {
				throw new System.Exception ("texture is null");
			}

			// Create the game object
			GameObject newObj = (GameObject)Object
				.Instantiate (sampleCube, sampleCube.transform.position, Quaternion.identity);

			if (!this.simplifiedProblem) {
				newObj.GetComponent<Renderer> ().materials [1].mainTexture = texture;
			}

			newObj.transform.localScale = new Vector3 (1.0f, 1.0f, 1.0f) * (float)blockSize;
			newObj.name = "Cube_Logo_" + ix;

			// Simplified problem adds a unique color to every block
			if (this.simplifiedProblem) {
				Material[] materials = newObj.GetComponent<Renderer> ().materials;
				materials [0].EnableKeyword ("_EMISSION");
				materials [0].SetColor ("_EmissionColor", this.colors [ix]);
			}

			newObj.SetActive (false);
			cubes.Add (newObj);
			ix++;
		}

		return cubes;
	}

	private List<GameObject> createDigitCubes(ObjectSystem system, GameObject sampleCube, GameObject sampleDigits) {

		this.colors = this.getUniqueColors ();
		this.digitText = new List<GameObject> ();
		List<GameObject> cubes = new List<GameObject> ();

		// Make the cube inactive
		sampleCube.SetActive (false);

		int ix = 0;
		float blockSize = (float)system.getSizeOfBlock ();

		Material[] newMaterial = new Material[1];
		newMaterial [0] = sampleCube.GetComponent<Renderer> ().materials [0];
		sampleCube.GetComponent<Renderer> ().materials = newMaterial;

		foreach (string brand in brandsList) {

			Debug.Log ("Creating cube for digit: " + (ix + 1));
			// Create the game object
			GameObject newObj = (GameObject)Object
				.Instantiate (sampleCube, sampleCube.transform.position, Quaternion.identity);

			if (!this.simplifiedProblem) {
				// Create digit and write the letter
				GameObject newDigits = (GameObject)Object
					.Instantiate (sampleDigits, sampleDigits.transform.position, sampleDigits.transform.rotation);
				newDigits.name = "Digits_" + ix;
				Vector3 oldScale = newDigits.transform.localScale;
				newDigits.GetComponent<TextMesh> ().text = (ix + 1).ToString ();
				newDigits.SetActive (false);
				this.digitText.Add (newDigits);
				newDigits.transform.parent = newObj.transform;
			}

			newObj.transform.localScale = new Vector3 (1.0f, 1.0f, 1.0f) * (float)blockSize;
			newObj.name = "Cube_Digit_" + ix;

			// Simplified problem adds a unique color to every block
			if (this.simplifiedProblem) {
				Material[] materials = newObj.GetComponent<Renderer> ().materials;
				materials [0].EnableKeyword ("_EMISSION");
				materials [0].SetColor ("_EmissionColor", this.colors [ix]);
			}

			ix++;

			newObj.SetActive (false);
			cubes.Add (newObj);
		}

		sampleDigits.SetActive (false);

		return cubes;
	}

	private void render(ObjectSystem system, Decoration pointDecoration) {

		List<Point3D> objectLocations = system.getObjectLocations ();

		for (int i = 0; i < brandsList.Count; i++) {
			this.logoCubes [i].SetActive (false); 
			this.digitCubes [i].SetActive (false);
			if (this.decoration == Decoration.Digits || this.decoration == Decoration.Both) {
				this.digitText [i].SetActive (false);
			}
		}

		int ix = 0;
		foreach (Point3D objectLocation in objectLocations) {

			Vector3 spawnPoint = new Vector3 ((float)objectLocation.getX (), 
				                     (float)objectLocation.getY (), 
				                     (float)objectLocation.getZ ());
			GameObject cubeObj = null;
			if (pointDecoration == Decoration.Logos) {
				cubeObj = this.logoCubes [ix];
			} else if (pointDecoration == Decoration.Digits) {
				cubeObj = this.digitCubes [ix];
			}

			cubeObj.transform.position = spawnPoint;
			cubeObj.transform.rotation = new Quaternion (0.0f, 0.0f, 0.0f, 0.0f);
			cubeObj.SetActive (true);

			if (pointDecoration == Decoration.Digits) {
				GameObject digitTex = this.digitText [ix];
				digitTex.transform.position = new Vector3 (spawnPoint.x, spawnPoint.y, spawnPoint.z);
				digitTex.SetActive (true);
			}

			ix++;
		}
	}

	// TEMPORARY: changes color of the block to be moved and moves the GoalState plane
	// to where the block has to be.
	private void highlightMainBlockAndGoal(ObjectSystem system, ObjectSystem toReach) {

		/*if (!this.simplifiedProblem) {
			return;
		}*/

		// Find the changed object
		List<Point3D> system1 = system.getObjectLocations();
		List<Point3D> system2 = toReach.getObjectLocations ();

		int i = 0;
		for(int k = 0; k < system1.Count; k++) {

			Point3D o1 = system1 [k];
			Point3D o2 = system2 [k];

			if (o1.calc2DL2Distance (o2) > 0.1f) {
				i = k;
				break;
			}
		}

		// Change all cube to their default color
		/*for(int j = 0; j < this.cubes.Count; j++) {
			GameObject cubeObj = this.cubes[j];
			Material[] materials = cubeObj.GetComponent<Renderer> ().materials;
			materials[0].SetColor ("_EmissionColor", this.colors[j]);
		}

		// Change the select cube to BLACK color
		GameObject selectedCubeObj = this.cubes [i];
		Material[] selectedCubeMaterials = selectedCubeObj.GetComponent<Renderer> ().materials;
		selectedCubeMaterials[0].SetColor ("_EmissionColor", Color.black);
		Vector3 blackPos = selectedCubeObj.transform.position;
		Debug.Log ("Black cube at " + blackPos.x + ", " + blackPos.y + ", " + blackPos.z);*/

		// Move plate to where the cubeObj has to be moved
		GameObject goalStatePlane = GameObject.Find("GoalState");
		Point3D finalPoint = system2 [i];
		goalStatePlane.transform.position = new Vector3((float)finalPoint.getX(),
			(float)finalPoint.getY(), (float)finalPoint.getZ());
	}

	// Shuffle algorithm based on Fisher-Yates algorithm
	private static void Shuffle(List<DataPoint> list) {  

		int n = list.Count;  
		while (n > 1) {  
			n--;  
			int k = InitScript.rnd.Next (n + 1);  
			DataPoint value = list [k];  
			list [k] = list [n];  
			list [n] = value;  
		}  
	}

	private List<DataPoint> readFromInstructionFile() {

		string[] instructions = System.IO.File.ReadAllLines ("Assets/inst.txt");
		Debug.Log ("Read  " + instructions.Length + "  many instructions ");

		for (int i = 0; i < instructions.Length; i++) {
			Debug.Log ("Instruction => " + instructions [i]);
		}

		List<DataPoint> newDataset = new List<DataPoint> ();

		for (int i = 0; i < instructions.Length; i++) {
			string inst_ = instructions [i].ToLower ();
			foreach (DataPoint dp in this.dataset) {
				string instMan = dp.getInstruction ().ToLower ();
				if (instMan.StartsWith (inst_)) {
					newDataset.Add (dp);
					break;
				}
			}
		}

		return newDataset;
	}

	// A test function used when running in simulator
	private void testLocally() {

		GameObject cube = GameObject.Find ("Cube");
		GameObject sampleDigits = GameObject.Find ("Digits");

		this.dataset = DatasetUtil.parseJson (this.dataFileName, this.decoration);

		System.IO.StreamWriter file = new System.IO.StreamWriter ("./training_instructions.txt");
		this.cachedTraj = new CachedTrajectory ();
		int datasetSize = this.dataset.Count;

		for (int i = 0; i < datasetSize; i++) {
			DataPoint dp = this.dataset [i];
			Episode eps = dp.getEpisode ();
			int startIndex = dp.getStartFrame ();
			int endIndex = dp.getEndFrame ();

			file.WriteLine (dp.getInstruction ());

			if (!this.cachedTraj.isExist (eps, startIndex, endIndex)) {
				ObjectSystem system1 = eps.getEnvironmentByIndex (startIndex);
				ObjectSystem toReach1 = eps.getEnvironmentByIndex (endIndex);
				TrajectoryResult trajResult = system1.findShortestPathAStar (toReach1);
				this.cachedTraj.addTrajectory (eps, startIndex, endIndex, trajResult);
			}

			List<int> actions = this.cachedTraj.getTrajectory (eps, startIndex, endIndex).getTrajectory ();
		}

		file.Flush ();
		file.Close ();
		return;

		/*Episode eps = null;
		int start = -1, end = -1;
		int flag = 0;
		Decoration pointDecoration = Decoration.Blank;
		string inst = "";
		int ctr = 0;
		for (int i = 0; i < this.dataset.Count; i++) {
			
			eps = this.dataset [i].getEpisode ();
			start = this.dataset [i].getStartFrame ();
			end = this.dataset [i].getEndFrame ();
			pointDecoration = this.dataset [i].getDecoration ();
			inst = this.dataset [i].getInstruction ();

			ObjectSystem system = eps.getEnvironmentByIndex (start);
			ObjectSystem toReach = eps.getEnvironmentByIndex (end);

			this.logoCubes = this.createLogoCubes (system, cube);
			this.digitCubes = this.createDigitCubes (system, cube, sampleDigits);

			GameObject ground = GameObject.Find ("Ground");
			ground.transform.localScale = new Vector3 (1.0f, 1.0f, 1.0f) * 0.25f;

			this.render (system, pointDecoration);
			this.highlightMainBlockAndGoal (system, toReach);

			Bounds bound = this.digitCubes[0].GetComponent<MeshFilter> ().mesh.bounds;
			Debug.Log ("Bound is " + bound.size.x);

			return;

			TrajectoryResult trajResult = system.findShortestPathAStar (toReach);
			List<int> actions = trajResult.getTrajectory ();
			string s = "";
			foreach (int action in actions) {
				int block = (int)(action / 4);
				int dir = action % 4;
				string dir_ = "";
				if (dir == 0) {
					dir_ = "west";//"north";
				} else if (dir == 1) {
					dir_ = "east";//"south";
				} else if (dir == 2) {
					dir_ = "north";//"east";
				} else if (dir == 3) {
					dir_ = "south";//"west";
				}

				s = s + block + "-" + dir_ + "; ";
			}
			Debug.Log ("Traj is " + s);
			Debug.Log ("Inst is  " + inst.ToLower ());
			Debug.Log ("Traj count " + actions.Count);

			//GameObject goalStatePlane = GameObject.Find ("GoalState");
			//goalStatePlane.SetActive (false);

			/Application.CaptureScreenshot ("Intro_Example.png");

			break;
		}

		float groundSizeX = GameObject.Find("Ground").GetComponent<Collider> ().bounds.size.x;
		float groundSizeZ = GameObject.Find("Ground").GetComponent<Collider> ().bounds.size.z;
		Debug.Log ("Size of groud is " + groundSizeX + " ,  " + groundSizeZ);
		return;*/
	}

	private bool isSeenDemonstration(List<SeenDemonstrations> seenDemonstrations, Episode eps, int start, int end) {
		foreach (SeenDemonstrations seen in seenDemonstrations) {
			if (seen.isSame (eps, start, end)) {
				return true;
			}
		}

		return false;
	}

	// Use this for initialization
	void Start () {

		InitScript.rnd = new System.Random (1234);

		System.IO.StreamReader file = new System.IO.StreamReader ("Assets/config.txt");
		string line1 = file.ReadLine ();
		string line2 = file.ReadLine ();
		string line3 = file.ReadLine ();
		string line4 = file.ReadLine ();
		string line5 = file.ReadLine ();
		string line6 = file.ReadLine ();
		string line7 = file.ReadLine ();
		string line8 = file.ReadLine ();
		string line9 = file.ReadLine ();
		string line10 = file.ReadLine ();
		string line11 = file.ReadLine ();
		string line12 = file.ReadLine ();

		int colon1 = line1.IndexOf (':');
		int colon2 = line2.IndexOf (':');
		int colon3 = line3.IndexOf (':');
		int colon4 = line4.IndexOf (':');
		int colon5 = line5.IndexOf (':');
		int colon6 = line6.IndexOf (':');
		int colon7 = line7.IndexOf (':');
		int colon8 = line8.IndexOf (':');
		int colon9 = line9.IndexOf (':');
		int colon10 = line10.IndexOf (':');
		int colon11 = line11.IndexOf (':');
		int colon12 = line12.IndexOf (':');

		this.numDatapoints = int.Parse (line1.Substring (colon1 + 1));
		this.shuffleBeforeSelect = bool.Parse (line2.Substring (colon2 + 1));
		this.dataFileName = line3.Substring (colon3 + 1);
		this.simplifiedProblem = bool.Parse (line4.Substring (colon4 + 1));
		this.horizon = int.Parse (line5.Substring (colon5 + 1));
		this.rewardFunctionType = int.Parse (line6.Substring (colon6 + 1));
		this.uselocalhost = bool.Parse (line7.Substring (colon7 + 1));
		this.stopActionReward = double.Parse (line8.Substring (colon8 + 1));
		this.screenSize = int.Parse (line9.Substring (colon9 + 1));
		string decorationSpecification = line10.Substring (colon10 + 1);
		if (decorationSpecification.CompareTo ("logo") == 0) {
			this.decoration = Decoration.Logos;
		} else if (decorationSpecification.CompareTo ("digit") == 0) {
			this.decoration = Decoration.Digits;
		} else if (decorationSpecification.CompareTo ("blank") == 0) { 
			Debug.LogError ("Not implemented decoration: " + decorationSpecification);
		} else if (decorationSpecification.CompareTo ("both") == 0) {
			this.decoration = Decoration.Both;
		} else {
			Debug.LogError ("Unknown decoration: " + decorationSpecification);
		}
		this.repeat = bool.Parse (line11.Substring (colon11 + 1));
		this.percentDemonstrations = float.Parse (line12.Substring (colon12 + 1));

		Debug.Log ("Num Data Points " + this.numDatapoints);
		Debug.Log ("Shuffle before select " + this.shuffleBeforeSelect);
		Debug.Log ("Data file name " + this.dataFileName);
		Debug.Log ("Simplified problem " + this.simplifiedProblem);
		Debug.Log ("Horizon " + this.horizon);
		Debug.Log ("Reward function type " + this.rewardFunctionType);
		Debug.Log ("Use localhost " + this.uselocalhost);
		Debug.Log ("Stop Action Reward " + this.stopActionReward);
		Debug.Log ("Screen size " + this.screenSize);
		Debug.Log ("Decoration is " + this.decoration);
		Debug.Log ("Percentage of demonstations to use is " + this.percentDemonstrations);

		/////////////
//		this.testLocally();
		/////////////

		Application.runInBackground = true;
		Screen.SetResolution (this.screenSize, this.screenSize, false);
	}

	void Update1() {

		if (this.first) {
			if (Screen.height != this.screenSize || Screen.width != this.screenSize) {
				return;
			}
			this.first = false;
		} else {
			return;
		}

		this.dataset = DatasetUtil.parseJson (this.dataFileName, this.decoration);

		this.cachedTraj = new CachedTrajectory ();
		int datasetSize = this.dataset.Count;

		System.IO.StreamWriter file = new System.IO.StreamWriter (@"datafile_" + this.dataFileName + ".txt");

		for (int i = 0; i < datasetSize; i++) {
			DataPoint dp = this.dataset [i];
			Episode eps = dp.getEpisode ();
			int startIndex = dp.getStartFrame ();
			int endIndex = dp.getEndFrame ();
			file.WriteLine (dp.getInstruction ());

			List<int> blocksMoved = dp.blocksMoved ();

			if (blocksMoved.Count != 1) {
				Debug.LogError ("Moved more than 1 block!!!");
			}

			int goldBlock = blocksMoved [0];
			Point3D pt = dp.getEpisode ().getEnvironmentByIndex (dp.getEndFrame ()).getObjectLocations () [goldBlock];
			string info = goldBlock + " " + pt.getX () + " " + pt.getY () + " " + pt.getZ ();
			file.WriteLine (info);

			if (!this.cachedTraj.isExist (eps, startIndex, endIndex)) {
				ObjectSystem system1 = eps.getEnvironmentByIndex (startIndex);
				ObjectSystem toReach1 = eps.getEnvironmentByIndex (endIndex);
				TrajectoryResult trajResult = system1.findShortestPathAStar (toReach1);
				this.cachedTraj.addTrajectory (eps, startIndex, endIndex, trajResult);
			}
		}

		file.Flush ();
		file.Close ();
			
		// Start with the environment in the training data
		this.datapointIndex = 0;
		DataPoint current = this.dataset [this.datapointIndex];
		int start = current.getStartFrame ();
		int end = current.getEndFrame ();
		Episode episode = current.getEpisode ();
		ObjectSystem system = episode.getEnvironmentByIndex (start);
		ObjectSystem toReach = episode.getEnvironmentByIndex (end);
		TrajectoryResult trajResult1 = this.cachedTraj.getTrajectory(episode, start, end);
		List<int> trajectory = trajResult1.getTrajectory ();
		List<Point3D> trajectoryPoints = trajResult1.getTrajectoryPoints ();
		string instruction = current.getInstruction ();
		Decoration pointDecoration = current.getDecoration ();

		Debug.Log ("Up and running");

		//Read the grid
		GameObject ground = GameObject.Find ("Ground");
		Bounds groundBounds = ground.GetComponent<MeshFilter> ().mesh.bounds;
		float groundWidth = groundBounds.size.x;
		float groundHeight = groundBounds.size.z;  

		Debug.Log ("Ground: Width " + groundWidth + " height " + groundHeight); 

		//Read the sample cube information
		GameObject cube = GameObject.Find ("Cube");
		Bounds cubeBounds = cube.GetComponent<MeshFilter> ().mesh.bounds;
		float cubeWidth = cubeBounds.size.x;
		float cubeHeight = cubeBounds.size.z;  

		Debug.Log ("Cube: Width " + cubeWidth + " height " + cubeHeight); 

		//Render the system
		this.logoCubes = this.createLogoCubes (system, cube);
		GameObject sampleDigits = GameObject.Find ("Digits");
		this.digitCubes = this.createDigitCubes (system, cube, sampleDigits);
		ground.transform.localScale = new Vector3 (1.0f, 1.0f, 1.0f) * 0.25f;
		this.render (system, pointDecoration);
		this.highlightMainBlockAndGoal (system, toReach);

		List<GameObject> cubes = null;
		if (pointDecoration == Decoration.Logos) {
			cubes = this.logoCubes;
		} else if (pointDecoration == Decoration.Digits) {
			cubes = this.digitCubes;
		}

		// Add robot controller
		GameObject robot = GameObject.Find ("Robot");
		robot.AddComponent<RobotController> ();
		RobotController robotController = robot.GetComponent<RobotController> ();
		robotController.init (cubes, (float)system.getSizeOfBlock (), this.stepSize, this.horizon);

		// Initialize the MDP Manager which computes rewards.
		double gamma = 0.1;
		double failureReward = -5;
		double winReward = 5;
		robotController.initMDPManager (gamma, failureReward, winReward, this.stopActionReward, 
			this.rewardFunctionType, this.simplifiedProblem, this.horizon);

		// Set the robot controller to work on the current problem
		robotController.changeEpisode (cubes, system, toReach, trajectory, trajectoryPoints, brandsList.Count, episode.getBlockSize ());

		// Add agent messenger for listening
		AgentMessenger.uselocalhost = this.uselocalhost;
		robot.AddComponent<AgentMessenger> ();
		this.agentMessenger = robot.GetComponent<AgentMessenger> ();

		// Attach the robot controller to agent messenger
		this.agentMessenger.attachRobotController (robotController);

		//Find trajectory
		string s = "";
		foreach (int i in trajectory) {
			s = s + i + ",";
		}

		// TODO Clearly define message protocol
		StartCoroutine (initConnection ("Reset#0#img/Screenshot.png#" + instruction +"#" + s));
	}

	void Update() {

		if (this.first) {
			if (Screen.height != this.screenSize || Screen.width != this.screenSize) {
				return;
			}
			this.first = false;
		} else {
			return;
		}
			
		this.dataset = DatasetUtil.parseJson (this.dataFileName, this.decoration);
		//this.dataset = this.readFromInstructionFile ();

		/// Remove pairs of replicated state-end environments
		/// in simplified problem
		if (this.simplifiedProblem) {
			List<DataPoint> newDataset = new List<DataPoint> ();

			int oldDatasetSize = this.dataset.Count;
			for (int i = 0; i < oldDatasetSize; i++) {
				DataPoint dp = this.dataset [i];
				Episode eps = dp.getEpisode ();
				int startIndex = dp.getStartFrame ();
				int endIndex = dp.getEndFrame ();

				bool seen = false;
				foreach (DataPoint newDp in newDataset) {
					if (newDp.getStartFrame () == startIndex &&
					    newDp.getEndFrame () == endIndex &&
					    newDp.getEpisode () == eps) {
						seen = true;
						break;
					}
				}

				if (!seen) {
					newDataset.Add (dp);
				}
			}

			this.dataset = newDataset;
			Debug.Log ("After removing commons. The dataset file " +
			this.dataFileName + " has size " + this.dataset.Count);
		}

		Debug.Log ("Total dataset size " + this.dataset.Count);
		if (this.numDatapoints == -1) {
			this.numDatapoints = this.dataset.Count;
		}

		// Separate tune data and add it later
		List<DataPoint> tuning = new List<DataPoint> ();

		HashSet<Episode> epsLogo = new HashSet<Episode> ();
		HashSet<Episode> epsDigit = new HashSet<Episode> ();

		for (int i = 0; i < this.dataset.Count; i++) {

			DataPoint dp = this.dataset [i];
			Decoration type = dp.getDecoration ();
			Episode eps = dp.getEpisode ();

			if (type == Decoration.Logos) {
				if (epsLogo.Contains (eps)) {
					tuning.Add (dp);
				} else if (epsLogo.Count < 2) {
					//Space to add more episodes
					epsLogo.Add (eps);
					tuning.Add (dp);
				}
			}

			if (type == Decoration.Digits) {
				if (epsDigit.Contains (eps)) {
					tuning.Add (dp);
				} else if (epsDigit.Count < 2) {
					// Space to add more episodes
					epsDigit.Add (eps);
					tuning.Add (dp);
				}
			}
		}

		this.cachedTraj = new CachedTrajectory ();
		int datasetSize = this.dataset.Count;

		for (int i = 0; i < datasetSize; i++) {
			DataPoint dp = this.dataset [i];
			Episode eps = dp.getEpisode ();
			int startIndex = dp.getStartFrame ();
			int endIndex = dp.getEndFrame ();
			if (!this.cachedTraj.isExist (eps, startIndex, endIndex)) {
				ObjectSystem system1 = eps.getEnvironmentByIndex (startIndex);
				ObjectSystem toReach1 = eps.getEnvironmentByIndex (endIndex);
				TrajectoryResult trajResult = system1.findShortestPathAStar (toReach1);
				this.cachedTraj.addTrajectory (eps, startIndex, endIndex, trajResult);
			}
		}

		// Remove the tuning data
		foreach (DataPoint dp in tuning) {
			this.dataset.Remove (dp);
		}
		Debug.Log ("Dataset: Tuning " + tuning.Count + " and train is " + this.dataset.Count);

		if (this.shuffleBeforeSelect) {
			InitScript.Shuffle (this.dataset);
		}

		// Add the tuning data in the beginning
		List<DataPoint> smallerDataset = new List<DataPoint> ();
		foreach (DataPoint dp in tuning) {
			smallerDataset.Add (dp);
		}
			
		//////////////////////////////////////
		List<SeenDemonstrations> uniqueDemonstration = new List<SeenDemonstrations>();

		foreach (DataPoint dp in this.dataset) {
			bool isSeen = this.isSeenDemonstration (uniqueDemonstration, dp.getEpisode (), dp.getStartFrame (), dp.getEndFrame ());
			if (!isSeen) {
				SeenDemonstrations seen = new SeenDemonstrations (dp.getEpisode (), dp.getStartFrame (), dp.getEndFrame ());
				uniqueDemonstration.Add (seen);
			}
		}

		// Number of demonstations to consider
		int toUse = (int)(this.percentDemonstrations * uniqueDemonstration.Count);
		Debug.Log ("Number of demonstrations to are " + uniqueDemonstration.Count + " using " + toUse);

		List<SeenDemonstrations> usedDemonstrations = new List<SeenDemonstrations>();
		for (int i = 0; i < toUse; i++) {
			usedDemonstrations.Add (uniqueDemonstration [i]);
		}

		Debug.Log ("Using " + usedDemonstrations.Count + " out of " + uniqueDemonstration.Count);

		List<DataPoint> dpsSeen = new List<DataPoint>();
		List<DataPoint> dpsUnseen = new List<DataPoint> ();

		foreach (DataPoint dp in this.dataset) {
			
			if (this.isSeenDemonstration (usedDemonstrations, dp.getEpisode (), dp.getStartFrame (), dp.getEndFrame ())) {
				dpsSeen.Add (dp);
			} else {
				dpsUnseen.Add (dp);
			}
		}

		Debug.Log ("Size of seen dataset is " + dpsSeen.Count + " and unseen is " + dpsUnseen.Count);
		int total = tuning.Count + dpsSeen.Count + dpsUnseen.Count;
		Debug.Log ("Total accounted points are " + total); 

		foreach (DataPoint dp in dpsSeen) {
			smallerDataset.Add (dp);
		}

		foreach (DataPoint dp in dpsUnseen) {
			smallerDataset.Add (dp);
		}

		Debug.Log ("Size of smaller dataset is " + smallerDataset.Count);
		this.numDatapointsWithDemonstrations = tuning.Count + dpsSeen.Count;
		Debug.Log ("MAGIC: Consider the first " + this.numDatapointsWithDemonstrations + " points ");
		//////////////////////////////////////

		/*foreach (DataPoint dp in this.dataset) {
			smallerDataset.Add (dp);
		}*/

		this.dataset = smallerDataset;

		// Start with the environment in the training data
		this.datapointIndex = 0;
		DataPoint current = this.dataset [this.datapointIndex];
		int start = current.getStartFrame ();
		int end = current.getEndFrame ();
		Episode episode = current.getEpisode ();
		ObjectSystem system = episode.getEnvironmentByIndex (start);
		ObjectSystem toReach = episode.getEnvironmentByIndex (end);
		TrajectoryResult trajResult1 = this.cachedTraj.getTrajectory (episode, start, end);
		List<int> trajectory = trajResult1.getTrajectory ();
		List<Point3D> trajectoryPoints = trajResult1.getTrajectoryPoints ();
		string instruction = current.getInstruction ();
		Decoration pointDecoration = current.getDecoration ();

		Debug.Log ("Up and running");

		//Read the grid
		GameObject ground = GameObject.Find ("Ground");
		Bounds groundBounds = ground.GetComponent<MeshFilter> ().mesh.bounds;
		float groundWidth = groundBounds.size.x;
		float groundHeight = groundBounds.size.z;  

		Debug.Log ("Ground: Width " + groundWidth + " height " + groundHeight); 

		//Read the sample cube information
		GameObject cube = GameObject.Find ("Cube");
		Bounds cubeBounds = cube.GetComponent<MeshFilter> ().mesh.bounds;
		float cubeWidth = cubeBounds.size.x;
		float cubeHeight = cubeBounds.size.z;  

		Debug.Log ("Cube: Width " + cubeWidth + " height " + cubeHeight); 

		//Render the system
		this.logoCubes = this.createLogoCubes (system, cube);
		GameObject sampleDigits = GameObject.Find ("Digits");
		this.digitCubes = this.createDigitCubes (system, cube, sampleDigits);
		ground.transform.localScale = new Vector3 (1.0f, 1.0f, 1.0f) * 0.25f;
		this.render (system, pointDecoration);
		this.highlightMainBlockAndGoal (system, toReach);

		List<GameObject> cubes = null;
		if (pointDecoration == Decoration.Logos) {
			cubes = this.logoCubes;
		} else if (pointDecoration == Decoration.Digits) {
			cubes = this.digitCubes;
		}

		// Add robot controller
		GameObject robot = GameObject.Find ("Robot");
		robot.AddComponent<RobotController> ();
		RobotController robotController = robot.GetComponent<RobotController> ();
		robotController.init (cubes, (float)system.getSizeOfBlock (), this.stepSize, this.horizon);

		// Initialize the MDP Manager which computes rewards.
		double gamma = 0.1;
		double failureReward = -5;
		double winReward = 5;
		robotController.initMDPManager (gamma, failureReward, winReward, this.stopActionReward, 
			this.rewardFunctionType, this.simplifiedProblem, this.horizon);

		// Set the robot controller to work on the current problem
		robotController.changeEpisode (cubes, system, toReach, trajectory, trajectoryPoints, brandsList.Count, episode.getBlockSize ());

		// Add agent messenger for listening
		AgentMessenger.uselocalhost = this.uselocalhost;
		robot.AddComponent<AgentMessenger> ();
		this.agentMessenger = robot.GetComponent<AgentMessenger> ();

		// Attach the robot controller to agent messenger
		this.agentMessenger.attachRobotController (robotController);

		//Find trajectory
		string s = "";
		foreach (int i in trajectory) {
			s = s + i + ",";
		}

		// TODO Clearly define message protocol
		StartCoroutine (initConnection ("Reset#0#img/Screenshot.png#" + instruction +"#" + s));
//		StartCoroutine (sendMessageWithGoalState ("Reset#0#img/Screenshot.png#" + instruction + "#" + s));
	}
	

	IEnumerator initConnection(string message) {

		GameObject goalStatePlane = GameObject.Find("GoalState");
		goalStatePlane.SetActive (false);

		yield return new WaitForEndOfFrame();

		Texture2D tex = new Texture2D(Screen.width, Screen.height, TextureFormat.RGBAFloat, false);
		tex.ReadPixels(new Rect(0f, 0f, Screen.width, Screen.height), 0, 0, false);
		tex.Apply();

		byte[] autoTex = tex.GetRawTextureData ();

		goalStatePlane.SetActive (true);

		//Send the image
		this.agentMessenger.sendMessage (autoTex);

		//Send the instruction
		this.agentMessenger.sendMessage (message);

		//Start listening to response from agent
		Thread initListening = new Thread (this.agentMessenger.listen);
		initListening.Start ();
	}

	IEnumerator sendMessageWithGoalState(string message) {

		GameObject goalStatePlane = GameObject.Find ("GoalState");
		goalStatePlane.SetActive (false);

		yield return new WaitForEndOfFrame ();

		Texture2D tex = new Texture2D (Screen.width, Screen.height, TextureFormat.RGBAFloat, false);
		tex.ReadPixels (new Rect (0f, 0f, Screen.width, Screen.height), 0, 0, false);
		tex.Apply ();

		byte[] autoTex = tex.GetRawTextureData ();

		goalStatePlane.SetActive (true);

		// Render to the goal image
		this.resetEnd ();

		StartCoroutine (addGoalStateAndSend (message, autoTex));
	}

	IEnumerator addGoalStateAndSend(string message, byte[] startState) {

		GameObject goalStatePlane = GameObject.Find ("GoalState");
		goalStatePlane.SetActive (false);

		yield return new WaitForEndOfFrame ();

		Texture2D tex = new Texture2D (Screen.width, Screen.height, TextureFormat.RGBAFloat, false);
		tex.ReadPixels (new Rect (0f, 0f, Screen.width, Screen.height), 0, 0, false);
		tex.Apply ();

		byte[] goalState = tex.GetRawTextureData ();

		goalStatePlane.SetActive (true);

		// Reset to the start image
		this.resetStart ();

		// Send the start image
		this.agentMessenger.sendMessage (startState);

		//Send the goal image
		this.agentMessenger.sendMessage (goalState);

		//Send the instruction
		this.agentMessenger.sendMessage (message);

		//Start listening to response from agent
		Thread initListening = new Thread (this.agentMessenger.listen);
		initListening.Start ();
	}

	public string reset() {

		if (this.repeat) {
			if (shownAlready) {
				// Jump to the next datapoint
				this.datapointIndex = (this.datapointIndex + 1) % this.dataset.Count;
				shownAlready = false;
			} else {
				// Shown the task again
				shownAlready = true;
			}
		} else {
			// Jump to the next datapoint
			this.datapointIndex = (this.datapointIndex + 1) % this.dataset.Count;
		}

		DataPoint current = this.dataset [this.datapointIndex];
		int start = current.getStartFrame ();
		int end = current.getEndFrame ();
		Episode episode = current.getEpisode ();
		ObjectSystem system = episode.getEnvironmentByIndex (start);
		ObjectSystem toReach = episode.getEnvironmentByIndex (end);
		Decoration pointDecoration = current.getDecoration ();
		this.render (system, pointDecoration);
		this.highlightMainBlockAndGoal (system, toReach);

		GameObject robot = GameObject.Find("Robot");
		RobotController robotController = robot.GetComponent<RobotController> ();

		/////////////////////////////////////
		Episode myEpisode = this.dataset [this.datapointIndex].getEpisode ();
		if (this.datapointIndex == 0) {
			// Part of seen demo. therefore
			// change the reward function to add demonstrations (type 4)
			robotController.changeRewardFunctionType (4);
		}

		if(this.datapointIndex == this.numDatapointsWithDemonstrations) {
			// Part of unseen demo. therefore 
			// change the reward function to remove demonstrations (type 2)
			robotController.changeRewardFunctionType (2);
		}
		/////////////////////////////////////

		TrajectoryResult trajResult1 = this.cachedTraj.getTrajectory(episode, start, end);
		List<int> trajectory = trajResult1.getTrajectory ();
		List<Point3D> trajectoryPoints = trajResult1.getTrajectoryPoints ();
		List<GameObject> cubes = null;
		if (pointDecoration == Decoration.Logos) {
			cubes = this.logoCubes;
		} else if (pointDecoration == Decoration.Digits) {
			cubes = this.digitCubes;
		}
		robotController.changeEpisode(cubes, system, toReach, trajectory, trajectoryPoints, brandsList.Count, episode.getBlockSize());

		string s = "";
		foreach (int i in trajectory) {
			s = s + i + ",";
		}

		return current.getInstruction () + "#" + s;
	}

	public void resetStart() {

		DataPoint current = this.dataset [this.datapointIndex];
		int start = current.getStartFrame ();
		int end = current.getEndFrame ();
		Episode episode = current.getEpisode ();
		ObjectSystem system = episode.getEnvironmentByIndex (start);
		ObjectSystem toReach = episode.getEnvironmentByIndex (end);
		Decoration pointDecoration = current.getDecoration ();
		this.render (system, pointDecoration);
		this.highlightMainBlockAndGoal (system, toReach);
	}

	public void resetEnd() {

		DataPoint current = this.dataset [this.datapointIndex];
		int start = current.getStartFrame ();
		int end = current.getEndFrame ();
		Episode episode = current.getEpisode ();
		ObjectSystem system = episode.getEnvironmentByIndex (start);
		ObjectSystem toReach = episode.getEnvironmentByIndex (end);
		Decoration pointDecoration = current.getDecoration ();
		this.render (toReach, pointDecoration);
		this.highlightMainBlockAndGoal (system, toReach);
	}
}

class SeenDemonstrations {

	private Episode eps;
	private int start, end;

	public SeenDemonstrations(Episode eps, int start, int end) {
		this.eps = eps;
		this.start = start;
		this.end = end;
	}

	public bool isSame(Episode eps, int start, int end) {
		if (this.eps == eps && this.start == start && this.end == end) {
			return true;
		} else {
			return false;
		}
	}
}

