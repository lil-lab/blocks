using System;
using UnityEngine;

public class TranslationStepAction {

	public enum Direction {North, South, East, West};

	private int cubeID;
	private Direction direction;
	private int actionID;

	public TranslationStepAction(int cubeID, Direction direction, int actionID) {
		this.cubeID = cubeID;
		this.direction = direction;
		this.actionID = actionID;
	}

	// Action description if of the form ID Direction e.g. "1 south", "1 north" etc.
	public static TranslationStepAction parse(string actionDescription) {

		string[] words = actionDescription.Split (new char[]{ ' ' });
		if (words.Length > 2) {
			throw new UnityException ("Expect the action as 'ID Direction' e.g., '1 north'. Found " + actionDescription);
		}

		int cubeID = int.Parse (words [0]);
		Direction direction;

		int directionID = 0;

		if (words [1].Equals ("north")) {
			direction = Direction.North;
			directionID = 0;
		} else if (words [1].Equals ("south")) {
			direction = Direction.South;
			directionID = 1;
		} else if (words [1].Equals ("east")) {
			direction = Direction.East;
			directionID = 2;
		} else if (words [1].Equals ("west")) {
			direction = Direction.West;
			directionID = 3;
		} else {
			throw new UnityException ("Expect direction to be 'north', 'south', 'east', 'west'. Found " + words[1]);
		}

		int actionID = cubeID * 4 + directionID;

		return new TranslationStepAction(cubeID, direction, actionID);
	}

	public int getCubeID() {
		return this.cubeID;
	}

	public Direction getDirection() {
		return this.direction;
	}

	public int getActionID() {
		return this.actionID;
	}
}

