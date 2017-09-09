using System;
using UnityEngine;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;

public class DatasetUtil {

	public static List<DataPoint> parseJson(string fileName, Decoration decoration) {

		List<DataPoint> dataset = new List<DataPoint> ();

		List<String> validDecorationString = new List<String> ();
		if (decoration == Decoration.Logos) {
			validDecorationString.Add ("logo");
		} else if (decoration == Decoration.Digits) {
			validDecorationString.Add ("digit");
		} else if (decoration == Decoration.Both) {
			validDecorationString.Add ("logo");
			validDecorationString.Add ("digit");
		}

		var json = System.IO.File.ReadAllText("Assets/" + fileName);
		JArray arr = JArray.Parse(json); // parse as array

		foreach (JObject root in arr) {

			// Check the decoration type
			string decorationType = root ["decoration"].ToString();
			bool success = false;
			Decoration pointDecoration = Decoration.Blank;

			foreach (string decorationString in validDecorationString) {
				if (decorationType.CompareTo (decorationString) == 0) {

					if(decorationString.CompareTo("digit") == 0) {
						pointDecoration = Decoration.Digits;
					} else if(decorationString.CompareTo("logo") == 0) {
						pointDecoration = Decoration.Logos;
					}

					success = true;
					break;
				}
			}

			if (!success) {
				continue;
			}

			// extract the block size
			float blockSize = float.Parse (root["side_length"].ToString());
			Debug.Log ("Block size " + blockSize);

			// Parse the list of environment
			List<ObjectSystem> systems = new List<ObjectSystem> ();
			JToken v = root ["states"];
			JToken envIt = v.First;
			while (envIt != null) {

				JToken it = envIt.First;
				List<Point3D> objectLocation = new List<Point3D> ();

				while (it != null) { //iterates over objects in the environment
					JToken u = it;
					JToken w = u.First; 
					double x = - double.Parse (w.ToString ());
					double y = double.Parse (w.Next.ToString ());
					double z = - double.Parse (w.Next.Next.ToString ());
					Debug.Log ("(" + x + ", " + y + ", " + z + ")\n");

					Point3D p = new Point3D (x, y, z);
					objectLocation.Add (p);

					it = it.Next;
				}

				Debug.Log ("Number of objects " + objectLocation.Count);
				ObjectSystem system = new ObjectSystem (blockSize, objectLocation);
				systems.Add (system);

				envIt = envIt.Next;
			}

			Debug.Log ("======= Done reading the first image  ======\n");
			Episode episode = new Episode (systems, blockSize);

			// Extract the sentences
			JToken frameIt = root ["notes"].First;
			while(frameIt != null) {

				int start = int.Parse (frameIt ["start"].ToString ());
				int finish = int.Parse (frameIt ["finish"].ToString ());
				string type = frameIt ["type"].ToString ();
				Type enumType = (Type) Enum.Parse(typeof(Type), type, false);

				Debug.Log("Start is " + start + " finish: " + finish + " type: " + type);

				if (enumType != Type.A0) {
					frameIt = frameIt.Next;
					continue;
				}

				JToken instructionIt = frameIt ["notes"].First;
				JToken userIt = frameIt ["users"].First;

				while (instructionIt != null) {

					if (userIt == null) {
						throw new ApplicationException ("Less user IDs than instructions");
					}

					Debug.Log ("Instruction: " + instructionIt);

					// Create a new datapoint
					String instruction = instructionIt.ToString();
					String userID = userIt.ToString();
					DataPoint datapoint = new DataPoint (instruction, enumType, start, finish, userID, episode, pointDecoration);
					dataset.Add (datapoint);

					instructionIt = instructionIt.Next;
					userIt = userIt.Next;
				}

				if (userIt != null) {
					throw new ApplicationException ("More user IDs than instructions");
				}

				frameIt = frameIt.Next;
			}
		}

		Debug.Log ("Dataset size " + dataset.Count);

		return dataset;
	}
}

