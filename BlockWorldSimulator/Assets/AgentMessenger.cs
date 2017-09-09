using System;
using System.Net;
using System.Net.Sockets;
using System.IO;
using System.Text;
using System.Threading;
using UnityEngine;
using System.Collections;
using RobotControllerFunctionality;

public class AgentMessenger : MonoBehaviour {
	
	// Part of code borrowed from MSDN C# docs: 
	// Client code: https://msdn.microsoft.com/en-us/library/kb5kfec7(v=vs.110).aspx
	// Server code: https://msdn.microsoft.com/en-us/library/6y0e13d3(v=vs.110).aspx

	// Incoming data from the client.
	public static string data = null;

	// Robot controller
	private RobotController robotController;
	private Socket listener;

	// Contains data received
	private byte[] bytes;

	// Handler for listening
	private Socket handler;

	// Number of jobs sent
	private int jobsSend;

	// Datastructures for reading image
	//private byte[] image;
	private Texture2D tex;

	// Message id
	private int msgID;

	// Reset flag
	// TODO replace this ugly hack
	private bool readyToReset, stopAction;

	// use local host
	public static bool uselocalhost;

	// Goal state
	private GameObject goalState;

	public AgentMessenger () {

		this.robotController = null;
		this.handler = null;
		this.readyToReset = false;
		this.stopAction = false;
		this.msgID = 0;

		// Initialize size of the image and image related datastructures
		this.tex = new Texture2D(Screen.width, Screen.height, TextureFormat.RGBAFloat, false);

		// Data buffer for incoming data.
		this.bytes = new Byte[1024];

		// Establish the local endpoint for the socket.
		// Dns.GetHostName returns the name of the 
		// host running the application.

		IPAddress ipAddress = null;
		int port = 11000;
		if(uselocalhost) {
			ipAddress = IPAddress.Any;
			port = 11000;
		} else {
			IPHostEntry ipHostInfo = Dns.GetHostEntry(Dns.GetHostName());
			ipAddress = ipHostInfo.AddressList[0];
			port = 0;
		}
		IPEndPoint localEndPoint = new IPEndPoint(ipAddress, port);

		// Create a TCP/IP socket.
		this.listener = new Socket(ipAddress.AddressFamily,
			SocketType.Stream, ProtocolType.Tcp );

		this.goalState = GameObject.Find ("GoalState");

		// Bind the socket to the local endpoint and 
		// listen for incoming connections.
		try {
			this.listener.Bind(localEndPoint);
			this.listener.Listen(10);
			int assignedPort = ((IPEndPoint)this.listener.LocalEndPoint).Port;
			Debug.Log ("Use localhost " + uselocalhost + ". Running at " + ipAddress.ToString () + " at port " + assignedPort);
		} catch (Exception e) {
			Debug.LogError( "Exception " + e.ToString());
		}
	}

	public void attachRobotController(RobotController robotController) {
		this.robotController = robotController;
	}

	void Update() {

		int jobs = this.robotController.getNumberOfJobsFinished ();

		if (jobs == 1 || this.stopAction) { // 1 should be replaced by number of jobs sent

			// TODO Should be handled by kernel in future

			// reset the jobs
			this.robotController.resetNumberOfJobsFinished ();

			// compute the reward
			double reward = this.robotController.calcReward (this.stopAction);
			string rewardStr = reward.ToString ();

			//string rewardStr = this.robotController.getRewardStr ();

//			StartCoroutine (this.sendResponse (reward));
			StartCoroutine (this.sendResponse (rewardStr));
		}

		if (this.readyToReset) {

			this.readyToReset = false;

			// Compute the Bisk Metric for this example before
			// resetting to another example.
//			double biskMetric = this.robotController.calcBiskMetric ();
//			double biskMetric = this.robotController.calcActionErrorMetric();
			double biskMetric = this.robotController.calcClosestApproachMetric();

			string resetFileNameAndInstruction = this.robotController.reset ();
			String msg = MessageHeader.Reset + "#" + biskMetric + "#" + resetFileNameAndInstruction;

			// Uncomment this line to work without sending the goal image
			StartCoroutine (this.sendResetResponse (msg));
//			StartCoroutine(this.sendResetResponseWithGoal(msg));
		}
	}

	IEnumerator sendResponse(string rewardStr) {

		this.goalState.SetActive (false);

		yield return new WaitForEndOfFrame();

		string fileName = "deprecated";

		// get message header
		MessageHeader messageHeader = this.robotController.getMessageHeader();

		// Check if reset need to be done
		bool shouldReset = this.robotController.shouldReset() || this.stopAction;

		string msg = null;

		if (shouldReset) {
			msg = messageHeader + "#" + rewardStr + "#" + fileName + "#reset"; 
			this.robotController.resetActionCounter ();
		} else {
			msg = messageHeader + "#" + rewardStr + "#" + fileName + "#"; 
		}

		this.stopAction = false;

		this.tex.ReadPixels(new Rect(0f, 0f, Screen.width, Screen.height), 0, 0, false);
		this.tex.Apply();
		byte[] ar = this.tex.GetRawTextureData ();
	
		this.goalState.SetActive (true);

		// Send the image
		this.sendMessage(ar);

		// Send message to agent
		this.sendMessage(msg);

		// Listen to message
		Thread th = new Thread(this.listen);
		th.Start ();
	}

	IEnumerator sendResetResponseWithGoal(string msg) {
		
		this.goalState.SetActive (false);

		yield return new WaitForEndOfFrame ();

		this.tex.ReadPixels (new Rect (0f, 0f, Screen.width, Screen.height), 0, 0, false);
		this.tex.Apply ();
		byte[] ar = this.tex.GetRawTextureData ();

		this.goalState.SetActive (true);

		// render the goal state
		this.robotController.resetEnd ();

		StartCoroutine (this.attachGoalAndSend (msg, ar));
	}

	IEnumerator attachGoalAndSend(string msg, byte[] startImage) {
		
		this.goalState.SetActive (false);

		yield return new WaitForEndOfFrame ();

		this.tex.ReadPixels (new Rect (0f, 0f, Screen.width, Screen.height), 0, 0, false);
		this.tex.Apply ();
		byte[] goalImage = this.tex.GetRawTextureData ();

		this.goalState.SetActive (true);

		// render back the start state
		this.robotController.resetStart ();

		// Send the start image
		this.sendMessage (startImage);

		// Send the goal image
		this.sendMessage (goalImage);

		// Send message to agent
		this.sendMessage (msg);

		// Listen for new message
		Thread th = new Thread (this.listen);
		th.Start ();
	}

	IEnumerator sendResetResponse(String msg) {

		this.goalState.SetActive (false);

		yield return new WaitForEndOfFrame ();

		this.tex.ReadPixels(new Rect(0f, 0f, Screen.width, Screen.height), 0, 0, false);
		this.tex.Apply();
		byte[] ar = this.tex.GetRawTextureData ();

		this.goalState.SetActive (true);

		this.sendMessage (ar);

		// Send message to agent
		this.sendMessage (msg);

		// Listen for new message
		Thread th = new Thread(this.listen);
		th.Start ();
	}

	public void listen() {

		try {
			if(this.handler == null) {
				Debug.Log ("Waiting for a connection...");
				// Program is suspended while waiting for an incoming connection.
				this.handler = this.listener.Accept();
			}

			data = null;

			// An incoming connection needs to be processed.
			while (true) {
				this.bytes = new byte[1024];
				int bytesRec = handler.Receive(this.bytes);
				data += Encoding.ASCII.GetString(this.bytes,0,bytesRec);
				if (data.IndexOf("<EOF>") > -1) {
					break;
				}
			}

			// Strip the data of EOF and add it to RobotController
			string[] words = data.Split(new string[] { "<EOF>" }, StringSplitOptions.None);

			// The message should either be an acknowledgement to reset
			// or should be a message for the continuing episode
			// TODO make this separation cleaner
			if (words[0].CompareTo("Ok-Reset") == 0) {
				// TODO currently this is thread safe as there
				// is no competing thread by design by later it may not be
				this.readyToReset = true;
			} else if (words[0].CompareTo("Stop") == 0) {
				// TODO currently this is thread safe as there
				// is no competing thread by design by later it may not be
				this.robotController.updateRewardString();
				this.stopAction = true;
			} else {
				string[] actionDescriptons = words;
				this.robotController.updateRewardString();
				for(int i = 0; i < actionDescriptons.Length - 1; i++) {
					this.robotController.addJob(actionDescriptons[i]);
				}
			}
		} catch(Exception e) {
			throw new ApplicationException ("Could not listen to message from agent. Error " + e);
		}
	}

	public void sendMessage(object message) {
		this.sendMessage ((String)message); 
	}

	public void sendMessage(String message) {

		try { 
			if(this.handler == null) {
				Debug.Log ("Waiting for a connection...");
				// Program is suspended while waiting for an incoming connection.
				this.handler = this.listener.Accept();
			}

			byte[] msg = Encoding.ASCII.GetBytes("Unity Manager: " + message);
			this.handler.Send(msg);
		} catch(Exception e) {
			throw new ApplicationException ("Could not send message to agent. Error " + e);
		}
	}

	public void sendMessage(byte[] message) {

		try { 
			if(this.handler == null) {
				Debug.Log ("Waiting for a connection...");
				// Program is suspended while waiting for an incoming connection.
				this.handler = this.listener.Accept();
			}

			this.msgID++;
			this.handler.Send(message);
		} catch(Exception e) {
			throw new ApplicationException ("Could not send message to agent. Error " + e);
		}
	}
}

