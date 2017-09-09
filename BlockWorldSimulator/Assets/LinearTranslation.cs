using UnityEngine;
using System.Collections;
using System.Collections.Generic; 
using System;

public class Job {

	private GameObject gameObj;
	private Vector3 targetPosition;

	public Job(GameObject gameObj, Vector3 targetPosition) {
		this.gameObj = gameObj;
		this.targetPosition = targetPosition;
	}

	public GameObject getGameObject() {
		return this.gameObj;
	}

	public Vector3 getTargetPosition() {
		return this.targetPosition;
	}
}

public class LinearTranslation : MonoBehaviour {

	private float speed;
	private bool collisionDetection;

	//Radius within which the objects are treated equal for practical purposes
	private float epsilon;
	private Queue<Job> jobQueue = null;
	private Vector3 targetPosition;

	//Target position is stale or not
	private bool stale;
	private Vector3 dir;

	// Tolerance value for float comparisong
	private float tol;

	//Last measured distance to target
	private double lastL2Distance;

	private SynchronizedInt numberOfJobsFinished = null;

	private MessageHeader messageHeader;

	private float groundOriginX, groundOriginZ;
	private float groundWidth, groundHeight;

	// Animate motion of blocks or simply jump them from one place to another.
	private bool animate;

	public void init(float speed) {
		this.jobQueue = new Queue<Job> ();
		this.speed = speed;
		this.epsilon = 0.001f;
		this.tol = 0.0001f;
		this.stale = true;
		this.animate = false;
		this.lastL2Distance = double.PositiveInfinity;
		this.numberOfJobsFinished = new SynchronizedInt ();

		GameObject ground = GameObject.Find("Ground");
		this.groundOriginX = ground.transform.position.x;
		this.groundOriginZ = ground.transform.position.z;

		Bounds groundBounds = ground.GetComponent<MeshFilter>().mesh.bounds;
		Vector3 groundScale = ground.transform.localScale;
		this.groundWidth = groundBounds.size.x * groundScale.x;
		this.groundHeight = groundBounds.size.z * groundScale.z;  
	}

	//Adds a job
	public void addJob(GameObject gameObj, Vector3 targetPosition) {

		// Add to queue
		Job newJob = new Job(gameObj, targetPosition);
		lock(this.jobQueue) {
			this.jobQueue.Enqueue (newJob);
		}
	}

	public bool isFinished() {
		if (this.jobQueue == null || this.jobQueue.Count == 0) {
			return true;
		} else {
			return false;
		}
	}

	public int getNumberOfJobsFinished() {
		return this.numberOfJobsFinished.getVal ();
	}

	public void resetNumberOfJobsFinished() {
		this.numberOfJobsFinished.reset ();
	}

	public void incrementNumberOfJobsFinished() {
		this.numberOfJobsFinished.increment ();
	}

	public MessageHeader getMessageHeader() {
		return this.messageHeader;
	}

	private double l2HorizontalDistance(GameObject obj, float xObj, float zObj) {

		float x = obj.transform.position.x;
		float z = obj.transform.position.z;

		return Mathf.Sqrt((x - xObj) * (x - xObj) + (z - zObj) * (z - zObj));
	}

	private bool checkForCollision(GameObject gameObj) {

		this.dir = this.targetPosition - gameObj.transform.position;
		this.dir.y = 0.0f;

		// Find perpendicular direction to dir in the xz-plane with unit norm
		// if dir = (x, 0, z) then this direction is given by 
		// - z/||dir|| + x/||dir||

		float magnitude = this.dir.magnitude;
		Vector3 perp = new Vector3 (-this.dir.z / magnitude, 0, this.dir.x / magnitude);

		// Get block size
		float boundSizeX = 0.1569f;
		float boundSizeZ = 0.1569f;
		if (false) {
			// For some hacky reason, get component failed in CB setting so hardcoding the constants.
			Bounds bound = gameObj.GetComponent<MeshFilter> ().mesh.bounds;
			boundSizeX = bound.size.x;
			boundSizeZ = bound.size.z;
		}

		Vector3 scale = gameObj.transform.localScale;
		float blockSizeX = boundSizeX * scale.x;
		float blockSizeZ = boundSizeZ * scale.z;

		float maxLength = Mathf.Max (blockSizeX, blockSizeZ);
		float extend = maxLength / (2.0f * magnitude) + this.tol;

		Vector3 adjustedDir = (1 + extend) * this.dir;
		float dist = adjustedDir.magnitude;

		// We shoot 3 raycast from center and two other points on the face of block
		// such that center and these two points are on straight line perpendicular
		// to direction of ray

		Vector3 adjustedPrep = perp * (maxLength / 3.0f);//replaced 2 by 3

		bool centerCollide = Physics.Raycast (gameObj.transform.position, this.dir, dist);

		if (centerCollide)
			return true;
		
		bool posCollide = Physics.Raycast (gameObj.transform.position + adjustedPrep, this.dir, dist);

		if (posCollide)
			return true;
		
		bool negCollide = Physics.Raycast (gameObj.transform.position - adjustedPrep, this.dir, dist);

		if (negCollide)
			return true;

		return false;
	}

	private bool fallsOffPlane(Vector3 targetPosition) {

		float x = targetPosition.x;
		float z = targetPosition.z;

		if (x > this.groundOriginX + 0.5 * this.groundWidth ||
		    x < this.groundOriginX - 0.5 * this.groundWidth ||
		    z > this.groundOriginZ + 0.5 * this.groundHeight ||
		    z < this.groundOriginZ - 0.5 * this.groundHeight) {
			return true;
		} else {
			return false;
		}
	}

	// Check if the point is inside pt
	private bool checkPointInside(Point3D pt, Point3D cube, float blockSize) {

		double tol = 0.002f;
		double half = blockSize / 2.0;
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
	private bool checkCollision(Point3D cube1, Point3D cube2, float blockSize) {

		double half = blockSize / 2.0;
		Point3D a = new Point3D (cube1.getX () + half, cube1.getY (), cube1.getZ () + half);
		Point3D b = new Point3D (cube1.getX () + half, cube1.getY (), cube1.getZ () - half);
		Point3D c = new Point3D (cube1.getX () - half, cube1.getY (), cube1.getZ () + half);
		Point3D d = new Point3D (cube1.getX () - half, cube1.getY (), cube1.getZ () - half);

		return this.checkPointInside (a, cube2, blockSize) || this.checkPointInside (b, cube2, blockSize)
			|| this.checkPointInside (c, cube2, blockSize) || this.checkPointInside (d, cube2, blockSize);
	}

	private bool checkCollisionBruteForce(Point3D cube, int blockId, List<Point3D> objLocs, float blockSize) {
	
		int ix = 0;
		foreach (Point3D obj in objLocs) {
			if (ix == blockId) {
				ix++;
				continue;
			} else {
				ix++;
			}

			if (this.checkCollision (cube, obj, blockSize)) {
				return true;
			}
		}

		return false;
	}

	// Executes a job quickly without using a queue
	public MessageHeader quickExec(GameObject gameObj, Vector3 direction, int blockId, List<Point3D> objLocs, float blockSize) {

		float x = gameObj.transform.position.x + direction.x;
		float y = gameObj.transform.position.y + direction.y;
		float z = gameObj.transform.position.z + direction.z;

		this.targetPosition = new Vector3 (x, y, z);

		// Check for collision
		//bool collides = this.checkForCollision (gameObj);
		Point3D cubePos = new Point3D (x, y, z);
		bool collides = checkCollisionBruteForce (cubePos, blockId, objLocs, blockSize);

		// Check whether moving the cube will make it fall off the ground
		bool fallsOffGround = this.fallsOffPlane (this.targetPosition);

		if (collides || fallsOffGround) {
			return MessageHeader.FailureACK;
		} else {
			return MessageHeader.SuccessACK;
		}
	}

	// Update is called once per frame
	void Update() {

		Job currentJob;

		lock (this.jobQueue) {

			if (this.jobQueue == null || this.jobQueue.Count == 0) {
				return;
			}

			currentJob = this.jobQueue.Peek ();
		}

		GameObject gameObj = currentJob.getGameObject ();

		if (this.stale) {
			Vector3 direction = currentJob.getTargetPosition ();
			float x = gameObj.transform.position.x + direction.x;
			float y = gameObj.transform.position.y + direction.y;
			float z = gameObj.transform.position.z + direction.z;

			this.targetPosition = new Vector3 (x, y, z);
			this.stale = false;
			this.lastL2Distance = double.PositiveInfinity;

			// TODO Check whether condition is already satisfied
			// some of our calculations depend on magnitude of diff
			// and can run into divide by 0 error
		}

		// Check for collision
		bool collides = this.checkForCollision(gameObj);

		// Check whether moving the cube will make it fall off the ground
		bool fallsOffGround = this.fallsOffPlane(this.targetPosition);

		if (!collides && !fallsOffGround) {
			if (this.animate) {
				float step = speed * Time.deltaTime;
				gameObj.transform.position = Vector3.MoveTowards (gameObj.transform.position, this.targetPosition, step);
			} else {
				gameObj.transform.position = new Vector3 (this.targetPosition.x, this.targetPosition.y, this.targetPosition.z);
			}
		}

		double distance = this.l2HorizontalDistance(gameObj, targetPosition.x, targetPosition.z);

		if (distance < this.epsilon ||
			distance >= this.lastL2Distance - (double)this.tol ||
			collides || fallsOffGround) {

			//Done.
			lock (this.jobQueue) {
				this.jobQueue.Dequeue ();
			}

			//Target position is no longer valid
			this.stale = true;
			this.incrementNumberOfJobsFinished ();

			//If the distance does not decrease then return failure
			if (distance < this.epsilon) {
				this.messageHeader = MessageHeader.SuccessACK;
			} else {
				this.messageHeader = MessageHeader.FailureACK;
			}
		}

		this.lastL2Distance = distance;
	}
}
