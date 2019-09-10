using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace MLAgents
{
    [DisallowMultipleComponent]
    public class WallContact : MonoBehaviour
    {
        [HideInInspector] public Agent agent;

        public bool touchingWall = false;
        private const string Wall = "climbingWall"; // Tag of ground object.
        public Rigidbody contact_rigidbody;
        /// <summary>
        /// Check for collision with ground, and optionally penalize agent.
        /// </summary>
        void OnCollisionEnter(Collision col)
        {
            if (col.transform.CompareTag(Wall))
            {
                contact_rigidbody = col.rigidbody;
                touchingWall = true;
            }
        }

        /// <summary>
        /// Check for end of wall collision and reset flag appropriately.
        /// </summary>
        void OnCollisionExit(Collision other)
        {
            if (other.transform.CompareTag(Wall))
            {
                touchingWall = false;
            }
        }
    }
}