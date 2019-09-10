using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace MLAgents
{
    [DisallowMultipleComponent]
    public class HoldContact : MonoBehaviour
    {
        [HideInInspector] public Agent agent;

        public bool touchingHold = false;
        private const string Hold = "Hold"; // Tag of ground object.
        public Rigidbody contact_rigidbody;
        public int hold_id = -1;
        public Vector3 contact_point;
        
        /// <summary>
        /// Check for collision with hold, and optionally penalize agent.
        /// </summary>
        void OnCollisionEnter(Collision col)
        {
            if (col.transform.CompareTag(Hold))
            {
                contact_rigidbody = col.rigidbody;
                contact_point = col.contacts[0].point;
                touchingHold = true;

                hold_id = col.transform.GetComponent<HoldInfo>().holdId;
            }
        }

        /// <summary>
        /// Check for end of hold collision and reset flag appropriately.
        /// </summary>
        void OnCollisionExit(Collision other)
        {
            if (other.transform.CompareTag(Hold))
            {
                touchingHold = false;
                hold_id = -1;
            }
        }
    }
}