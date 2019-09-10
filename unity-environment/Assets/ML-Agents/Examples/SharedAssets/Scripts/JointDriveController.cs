using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace MLAgents
{
    /// <summary>
    /// Used to store relevant information for acting and learning for each body part in agent.
    /// </summary>
    [System.Serializable]
    public class BodyPart
    {
        [Header("Body Part Info")] [Space(10)] public ConfigurableJoint joint;
        public Rigidbody rb;
        [HideInInspector] public Vector3 startingPos;
        [HideInInspector] public Quaternion startingRot;

        [Header("Ground & Target Contact")] [Space(10)]
        public GroundContact groundContact;
        public WallContact wallContact;
        public HoldContact holdContact;
        public TargetContact targetContact;

        [HideInInspector] public JointDriveController thisJDController;

        [Header("Current Joint Settings")] [Space(10)]
        public Vector3 currentEularJointRotation;

        [HideInInspector] public float currentStrength;
        public float currentXNormalizedRot;
        public float currentYNormalizedRot;
        public float currentZNormalizedRot;

        [Header("Other Debug Info")] [Space(10)]
        public Vector3 currentJointForce;

        public float currentJointForceSqrMag;
        public Vector3 currentJointTorque;
        public float currentJointTorqueSqrMag;
        public AnimationCurve jointForceCurve = new AnimationCurve();
        public AnimationCurve jointTorqueCurve = new AnimationCurve();

        /// <summary>
        /// Reset body part to initial configuration.
        /// </summary>
        public void Reset()
        {
            this.rb.transform.position = this.startingPos;
            this.rb.transform.rotation = this.startingRot;
            this.rb.velocity = Vector3.zero;
            this.rb.angularVelocity = Vector3.zero;
            if (this.groundContact)
            {
                this.groundContact.touchingGround = false;
            }

            if (this.wallContact)
            {
                this.wallContact.touchingWall = false;
            }

            if (this.holdContact)
            {
                this.holdContact.touchingHold = false;
                this.holdContact.hold_id = -1;
            }

            if (this.targetContact)
            {
                this.targetContact.touchingTarget = false;
            }
        }
        
        public void AddMinMaxTargetRotationValues(List<Vector2> MinMaxValues, int dof)
        {
            if (dof >= 1)
            {
                MinMaxValues.Add(new Vector2(joint.lowAngularXLimit.limit, joint.highAngularXLimit.limit));
            }
            if (dof >= 2)
            {
                MinMaxValues.Add(new Vector2(-joint.angularYLimit.limit, joint.angularYLimit.limit));
            }
            if (dof >= 3)
            {
                MinMaxValues.Add(new Vector2(-joint.angularZLimit.limit, joint.angularZLimit.limit));
            }
        }

        public void AddMinMaxStrengthValues(List<Vector2> MinMaxValues)
        {
            MinMaxValues.Add(new Vector2(0, thisJDController.maxJointForceLimit));
        }

        public Quaternion getJointCurrentRotation()
        {
            Quaternion rotated_q_child = Quaternion.Inverse(startingRot) * rb.transform.rotation;
            Quaternion rotated_q_parent = Quaternion.Inverse(thisJDController.bodyPartsDict[joint.connectedBody.transform].startingRot) * joint.connectedBody.transform.rotation;
            return rotated_q_child * Quaternion.Inverse(rotated_q_parent);
        }
        
        /// <summary>
        /// Apply torque according to defined goal `x, y, z` angle and force `strength`.
        /// </summary>
        public void SetJointTargetRotation(float x, float y, float z)
        {
            x = (x + 1f) * 0.5f;
            y = (y + 1f) * 0.5f;
            z = (z + 1f) * 0.5f;

            var xRot = Mathf.Lerp(joint.lowAngularXLimit.limit, joint.highAngularXLimit.limit, x);
            var yRot = Mathf.Lerp(-joint.angularYLimit.limit, joint.angularYLimit.limit, y);
            var zRot = Mathf.Lerp(-joint.angularZLimit.limit, joint.angularZLimit.limit, z);
            
            SetJointTargetRotationTrueVal(xRot, yRot, zRot);
        }

        public void SetJointStrength(float strength)
        {
            var rawVal = (strength + 1f) * 0.5f * thisJDController.maxJointForceLimit;
            SetJointStrengthTrueVal(rawVal);
        }

        public void SetJointTargetRotationTrueVal(float xRot, float yRot, float zRot)
        {
            //float maxDiff = 2 * Mathf.Rad2Deg * Mathf.PI * Time.fixedDeltaTime;

            currentXNormalizedRot = Mathf.InverseLerp(joint.lowAngularXLimit.limit, joint.highAngularXLimit.limit, xRot);
            currentYNormalizedRot = Mathf.InverseLerp(-joint.angularYLimit.limit, joint.angularYLimit.limit, yRot);
            currentZNormalizedRot = Mathf.InverseLerp(-joint.angularZLimit.limit, joint.angularZLimit.limit, zRot);

            joint.targetRotation = Quaternion.Euler(xRot, yRot, zRot);
            //Quaternion cq = getJointCurrentRotation();
            
            //float diff = Quaternion.Angle(cq, joint.targetRotation);
            //if (diff > maxDiff)
            //    joint.targetRotation = Quaternion.Slerp(cq, joint.targetRotation, maxDiff / diff);
            currentEularJointRotation = new Vector3(xRot, yRot, zRot);
        }

        public void SetJointStrengthTrueVal(float rawVal)
        {
            var jd = new JointDrive
            {
                positionSpring = thisJDController.maxJointSpring,
                positionDamper = thisJDController.jointDampen,
                maximumForce = rawVal
            };
            joint.slerpDrive = jd;
            currentStrength = jd.maximumForce;
        }
    }
    
    public class JointDriveController : MonoBehaviour
    {
        [Header("Joint Drive Settings")] [Space(10)]
        public float maxJointSpring;

        public float jointDampen;
        public float maxJointForceLimit;
        float facingDot;

        [HideInInspector] public Dictionary<Transform, BodyPart> bodyPartsDict = new Dictionary<Transform, BodyPart>();

        [HideInInspector] public List<BodyPart> bodyPartsList = new List<BodyPart>();

        /// <summary>
        /// Create BodyPart object and add it to dictionary.
        /// </summary>
        
        public void SetupBodyPart(Transform t)
        {
            BodyPart bp = new BodyPart
            {
                rb = t.GetComponent<Rigidbody>(),
                joint = t.GetComponent<ConfigurableJoint>(),
                startingPos = t.position,
                startingRot = t.rotation
            };
            bp.rb.maxAngularVelocity = 100;

            // Add & setup the ground contact script
            bp.groundContact = t.GetComponent<GroundContact>();
            if (!bp.groundContact)
            {
                bp.groundContact = t.gameObject.AddComponent<GroundContact>();
            }
            bp.groundContact.agent = gameObject.GetComponent<Agent>();
            
            bp.wallContact = t.GetComponent<WallContact>();
            if (!bp.wallContact)
            {
                bp.wallContact = t.gameObject.AddComponent<WallContact>();
            }
            bp.wallContact.agent = gameObject.GetComponent<Agent>();
            
            bp.holdContact = t.GetComponent<HoldContact>();
            if (!bp.holdContact)
            {
                bp.holdContact = t.gameObject.AddComponent<HoldContact>();
            }
            bp.holdContact.agent = gameObject.GetComponent<Agent>();
            
            // Add & setup the target contact script
            bp.targetContact = t.GetComponent<TargetContact>();
            if (!bp.targetContact)
            {
                bp.targetContact = t.gameObject.AddComponent<TargetContact>();
            }

            bp.thisJDController = this;
            bodyPartsDict.Add(t, bp);
            bodyPartsList.Add(bp);
        }

        public void GetCurrentJointForces()
        {
            foreach (var bodyPart in bodyPartsDict.Values)
            {
                if (bodyPart.joint)
                {
                    bodyPart.currentJointForce = bodyPart.joint.currentForce;
                    bodyPart.currentJointForceSqrMag = bodyPart.joint.currentForce.magnitude;
                    bodyPart.currentJointTorque = bodyPart.joint.currentTorque;
                    bodyPart.currentJointTorqueSqrMag = bodyPart.joint.currentTorque.magnitude;
                    if (Application.isEditor)
                    {
                        if (bodyPart.jointForceCurve.length > 1000)
                        {
                            bodyPart.jointForceCurve = new AnimationCurve();
                        }

                        if (bodyPart.jointTorqueCurve.length > 1000)
                        {
                            bodyPart.jointTorqueCurve = new AnimationCurve();
                        }

                        bodyPart.jointForceCurve.AddKey(Time.time, bodyPart.currentJointForceSqrMag);
                        bodyPart.jointTorqueCurve.AddKey(Time.time, bodyPart.currentJointTorqueSqrMag);
                    }
                }
            }
        }
    }
}
