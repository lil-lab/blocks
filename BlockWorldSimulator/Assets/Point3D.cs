using System;

public class Point3D {

	private readonly double x, y, z;

	public Point3D(double x, double y, double z) {
		this.x = x;
		this.y = y;
		this.z = z;
	}

	public double getX() {
		return this.x;
	}

	public double getY() {
		return this.y;
	}

	public double getZ() {
		return this.z;
	}

	// Adds another point3d and returns it
	public Point3D add(Point3D other) {
		return new Point3D (this.x + other.x, this.y + other.y, this.z + other.z);
	}

	public double calcL1Distance(Point3D other) {
		return Math.Abs (this.x - other.x) + Math.Abs (this.y - other.y) + Math.Abs (this.z - other.z);
	}


	public double calcL2Distance(Point3D other) {

		return Math.Sqrt((this.x - other.x) * (this.x - other.x) 
			           + (this.y - other.y) * (this.y - other.y) 
			           + (this.z - other.z) * (this.z - other.z));
	}

	public double calc2DL2Distance(Point3D other) {

		return Math.Sqrt((this.x - other.x) * (this.x - other.x) 
			+ (this.z - other.z) * (this.z - other.z));
	}

	public Point3D sub(Point3D other) {
		return new Point3D (this.x - other.x, this.y - other.y, this.z - other.z);
	}

	public double cosine2D(Point3D other) {
		double dotProduct = this.x * other.x + this.z * other.z;
		double l2This = this.norm2D ();
		double l2Other = other.norm2D ();
		double cosineDistance = dotProduct/(l2This * l2Other);

		return cosineDistance;
	}

	public double norm2D() {
		return Math.Sqrt (this.x * this.x + this.z * this.z);
	}

	public bool isColliding(Point3D other) {
		//TODO need to improve this function
		double dist = this.calc2DL2Distance (other);
		if (dist < 0.001) {
			return true;
		}
		return false;
	}
		
	public override string ToString() {
		return this.x + ", " + this.y + ", " + this.z;
	}
}