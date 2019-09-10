using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HitTarget : MonoBehaviour {
    
    // Use this for initialization
    void Start () {
		
	}
	
	// Update is called once per frame
	void Update () {
		
	}

    public int hit_key_id = -1;
    private void OnCollisionEnter(Collision collision)
    {
        if (collision.transform.tag == "target")
        {
            hit_key_id = collision.transform.GetComponent<HoldInfo>().holdId;
        }
    }

    private void OnCollisionExit(Collision collision)
    {
        hit_key_id = -1;
    }
}
