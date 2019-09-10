using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MoveHolds : MonoBehaviour {

    public int obstacle_index = -1;
	// Use this for initialization
	void Start () {
		
	}
	
	// Update is called once per frame
	void Update () {
		
	}
    
    private void MoveCollision(Collider collision)
    {
        if (collision.transform.tag == "Hold")
        {
            Vector3 p = this.transform.localPosition;
            Vector3 dis = collision.transform.localPosition - p;

            Vector3 xDir = this.transform.rotation * Vector3.right;
            Vector3 yDir = this.transform.rotation * Vector3.up;
            Vector3 zDir = this.transform.rotation * (-Vector3.forward);

            float nX, nY, nZ;

            nY = (dis.x * (-xDir.y) + dis.y * xDir.x) / (1e-6f + (yDir.x * (-xDir.y) + xDir.x * yDir.y));
            nX = (dis.x - nY * yDir.x) / (1e-6f + (xDir.x));
            nZ = (dis.z) / (zDir.z);

            float nXV, nYV, nZV;
            nXV = Mathf.Abs(nX / this.transform.localScale.x);
            nYV = Mathf.Abs(nY / this.transform.localScale.y);
            nZV = Mathf.Abs(nZ / this.transform.localScale.z);

            collision.transform.GetComponent<HoldInfo>().obstacleIndex = obstacle_index;

            if (nXV > nYV + 0.1f && nXV > nZV + 0.25f)
            {
                float dotProduct = Mathf.Sign(Vector3.Dot(dis, xDir));
                dis = dis + (dotProduct * this.transform.localScale.x / 2f - nX) * xDir;
                collision.transform.localPosition = p + dis;
                collision.transform.localPosition = new Vector3(collision.transform.localPosition.x, collision.transform.localPosition.y, this.transform.localPosition.z - 0.5f);
            }
            else if (nYV > nZV + 0.25f)
            {
                float dotProduct = Mathf.Sign(Vector3.Dot(dis, yDir));
                dis = dis + (dotProduct * this.transform.localScale.y / 2f - nY) * yDir;
                collision.transform.localPosition = p + dis;
                collision.transform.localPosition = new Vector3(collision.transform.localPosition.x, collision.transform.localPosition.y, this.transform.localPosition.z - 0.5f);
            }
            else
            {
                float dotProduct = Mathf.Sign(Vector3.Dot(dis, zDir));
                dis = dis + (dotProduct * this.transform.localScale.z / 2f - nZ) * zDir;
                collision.transform.localPosition = p + dis;
            }
        }
    }

    private void OnCollisionStay(Collision collision)
    {
        MoveCollision(collision.collider);
    }

    private void OnTriggerEnter(Collider collision)
    {
        MoveCollision(collision);
    }

    private void OnTriggerStay(Collider collision)
    {
        MoveCollision(collision);
    }

    private void OnCollisionEnter(Collision collision)
    {
        MoveCollision(collision.collider);
    }
}
