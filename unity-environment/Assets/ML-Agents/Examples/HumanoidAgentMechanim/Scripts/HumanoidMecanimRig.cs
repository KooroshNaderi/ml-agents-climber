using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using MLAgents;
using System.Runtime.InteropServices;
using Assets.ML_Agents.Examples.SharedAssets.Scripts;
//namespace AaltoGames{
//    //The MecanimODERig class is work-in-progress and contains a lot of legacy code.
//    //However, this interface should stay relatively stable - use it and avoid using any of the other methods.
public interface IPhysicsHumanoidRig
{
    //returns the number of control DOFs
    int numControlDOFs();
    float[] angleMinLimits();
    float[] angleMaxLimits();
    //Sets joint motor velocities such that the joint angles will reach the target after timeStep
    void driveToPose(float[] targetAngles, int sIndex = 0);//, float timeStep);
    //Returns the resulting effort after calling driveToPose() and simulating a step
    float getControlEffort();
    Vector3 COM();
    Vector3 COMVel();
    Vector3 pos(HumanBodyBones bone);
    Vector3 vel(HumanBodyBones bone);
    Vector3 avel(HumanBodyBones bone);
    Quaternion q(HumanBodyBones bone);
    bool loadState(int _slotIndex);
    bool saveState(int _freeSlotIdx);
}
public class HumanoidMecanimRig : MonoBehaviour, IPhysicsHumanoidRig
{
    public class BonelessTransformData
    {
        public Vector3 localPosition;
        public Quaternion localRotation;
    }

    public class BoneData
    {
        /// <summary>
        /// public functions
        /// </summary>

        public float GetCurrentAngleAxis(int _axis)
        {
            Quaternion _fromToRot = controlTransform.localRotation;

            Vector3 eulerAngle = GetAngleInPIReg(_fromToRot.eulerAngles);

            if (numFreedoms == 1)
            {
                float out_angle = Quaternion.Angle(parent.controlTransform.rotation, controlTransform.rotation);
                out_angle *= Mathf.Sign(Vector3.Dot(GetLocalAxis(_axis), eulerAngle));//hinge.angle;
                return out_angle;
            }
            
            return Vector3.Dot(GetLocalAxis(_axis), -eulerAngle);
        }

        public float GetTorqueJointAxis(int _axis)
        {
            return Vector3.Dot(GetTorqueJoint(), GetLocalAxis(_axis));
        }

        public float SetJointStrength(float rawVal)
        {
            if (amotor != null)
            {
                var jd = new JointDrive
                {
                    positionSpring = HumanoidMecanimRig.maxJointSpring,
                    positionDamper = HumanoidMecanimRig.jointDampen,
                    maximumForce = rawVal
                };
                amotor.slerpDrive = jd;
                currentStrength = rawVal;
                return rawVal;
            }
            else if (hinge != null)
            {
                JointSpring _hingeSpring = hinge.spring;
                _hingeSpring.spring = rawVal / 25f;
                _hingeSpring.damper = HumanoidMecanimRig.jointDampen;
                hinge.spring = _hingeSpring;

                currentStrength = rawVal;
                return rawVal;
            }

            currentStrength = 0f;
            return 0;
        }

        public Vector3 GetTorqueJoint()
        {
            if (numFreedoms == 1)
            {
                if (hinge != null)
                    return hinge.currentTorque;
            }
            else
            {
                if (amotor != null)
                    return amotor.currentTorque;
            }

            return new Vector3(0.0f, 0.0f, 0.0f);
        }

        public Vector3 GetForceJoint()
        {
            if (numFreedoms == 1)
            {
                if (hinge != null)
                    return hinge.currentForce;
            }
            else
            {
                if (amotor != null)
                    return amotor.currentForce;
            }

            return new Vector3(0.0f, 0.0f, 0.0f);
        }

        public void SetAMotorAngles(float xAngle, float yAngle, float zAngle)
        {
            if (amotor == null)
                return;

            if (amotor.slerpDrive.positionDamper > amotor.slerpDrive.positionSpring)
                amotor.targetAngularVelocity = new Vector3(xAngle, yAngle, zAngle);
            else
            {
                Vector3 angle = -xAngle * GetLocalAxis(0);
                angle += -yAngle * GetLocalAxis(1);
                angle += -zAngle * GetLocalAxis(2);
                amotor.configuredInWorldSpace = true;
                amotor.SetTargetRotation(Quaternion.Euler(angle), initialRotation);
            }

            return;
        }

        public void SetHingeAngle(float xAngle)
        {
            if (hinge == null)
                return;

            JointSpring hingeSpring = hinge.spring;
            hingeSpring.targetPosition = xAngle;

            hinge.spring = hingeSpring;

            return;
        }

        /// <summary>
        /// variables
        /// </summary>
        public GroundContact groundContact;
        public WallContact wallContact;
        public HoldContact holdContact;
        public TargetContact targetContact;
        //
        public float currentXNormalizedRot = 0;
        public float currentYNormalizedRot = 0;
        public float currentZNormalizedRot = 0;
        public float currentStrength = 0;
        //
        public Vector3 currentJointForce = new Vector3();
        public float currentJointForceSqrMag = 0f;
        public Vector3 currentJointTorque = new Vector3();
        public float currentJointTorqueSqrMag = 0f;
        //parent bone data
        public BoneData parent = null;
        public Transform parentTransform = null;
        //this bone data
        public Transform controlTransform = null;
        public GameObject controlObject = null;
        public HumanBodyBones mecanimBone;
        public Rigidbody rb = null;
        public Collider c = null;
        public float boneSize = 1.0f;
        public int index;
        public int numFreedoms = 3;
        public Vector3 boneVector;
        public float thickness;
        public string name;
        public float cachedMass;
        public Quaternion initialRotation = Quaternion.identity;
        // attached joint info
        public JointDefinition jointDef;
        public Joint joint = null;
        public Joint motor = null;
        public HingeJoint hinge = null;
        public ConfigurableJoint amotor = null;
        public int firstAngleIndex; //index of first angle of the motor in a vector of all motor angles (total getRigNumFreedoms() elements)
        public float FmaxScale = 1.0f;
        //relative path info
        public string relativeControlPath = "";
        //building Hierarchy
        public Transform[] transformHierarchyToParent;
        public Vector3[] localPositionHierarchyToParent;
        public Quaternion[] localRotationHierarchyToParent;

        /// <summary>
        /// private functions
        /// </summary>
        Vector3 GetLocalAxis(int _axisNum)
        {
            Vector3 _axis = new Vector3(1f, 0f, 0f);

            if (amotor != null)
            {
                switch (_axisNum)
                {
                    case 0:
                        _axis = amotor.axis;
                        break;
                    case 1:
                        _axis = amotor.secondaryAxis;
                        break;
                    case 2:
                        _axis = Vector3.Cross(amotor.axis, amotor.secondaryAxis);
                        break;
                    default:
                        break;
                }
                return _axis;
            }
            else if (hinge != null)
            {
                return hinge.axis;
            }
            return _axis;
        }

        Vector3 GetAxis(int _axisNum)
        {
            return controlTransform.rotation * GetLocalAxis(_axisNum);
        }

        float GetAngleInPIReg(float cAngle)
        {
            float cA = cAngle * Mathf.Deg2Rad;
            if (cA >= Mathf.PI)
            {
                cA -= 2 * Mathf.PI;
            }
            if (cA < -Mathf.PI)
            {
                cA += 2 * Mathf.PI;
            }

            return cA * Mathf.Rad2Deg;
        }

        Vector3 GetAngleInPIReg(Vector3 cAngle)
        {
            Vector3 retAng = new Vector3();
            for (int i = 0; i < 3; i++)
            {
                retAng[i] = GetAngleInPIReg(cAngle[i]);
            }
            return retAng;
        }
    }

    public class JointDefinition
    {
        public enum Type
        {
            BallAndSocket,
            Hinge,
            Slider,
            Universal,
            Hinge2,
            PrismaticAndRotoide,
            PrismaticUniversal,
            Piston,
            Fixed,
            AngularMotor,
            LinearMotor,
            Plane2D
        }

        public Joint Create(Transform startTransform, Rigidbody connectedBody)
        {
            Joint joint = null;
            switch (type)
            {
                case Type.Hinge:
                    joint = CreateHinge(startTransform, connectedBody);
                    break;
                case Type.BallAndSocket:
                    joint = CreateBallAndSocketWithAMotor(startTransform, connectedBody);
                    break;
                default:
                    Debug.LogError("Could not create joint for definition of type: " + type.ToString());
                    break;
            }

            return joint;
        }

        #region Hinge

        public static JointDefinition CreateHinge(float loStop = float.NegativeInfinity,
            float hiStop = float.PositiveInfinity)
        {
            JointDefinition def = new JointDefinition();
            def.type = Type.Hinge;
            def.lostop = loStop;
            def.histop = hiStop;
            return def;
        }

        private HingeJoint CreateHinge(Transform tr, Rigidbody body2)
        {
            if (tr == null)
                return null;

            HingeJoint joint = tr.gameObject.GetComponent<HingeJoint>();
            if (joint == null)
            {
                joint = tr.gameObject.AddComponent<HingeJoint>();
            }
            joint.connectedBody = body2;

            if (axis.HasValue)
                joint.axis = axis.Value;
            else
                joint.axis = joint.transform.forward;

            joint.useLimits = true;
            JointLimits _hingeLimit = joint.limits;
            _hingeLimit.min = this.lostop;
            _hingeLimit.max = this.histop;
            joint.limits = _hingeLimit;

            joint.useSpring = true;
            JointSpring _hingeSpring = joint.spring;
            _hingeSpring.spring = kp;
            _hingeSpring.damper = kd;
            joint.spring = _hingeSpring;

            JointMotor _hingeMotor = joint.motor;
            _hingeMotor.force = maxForce;
            joint.motor = _hingeMotor;
            joint.useMotor = false;

            joint.enablePreprocessing = false;
            
            return joint;
        }

        #endregion

        #region Ball And Socket
        public static JointDefinition CreateBallAndSocket(Vector3? loStop, Vector3? hiStop)
        {
            JointDefinition def = new JointDefinition();
            def.type = Type.BallAndSocket;

            if (loStop.HasValue)
            {
                Vector3 v = loStop.Value;
                def.lostop = v.x;
                def.lostop2 = v.y;
                def.lostop3 = v.z;
            }
            if (hiStop.HasValue)
            {
                Vector3 v = hiStop.Value;
                def.histop = v.x;
                def.histop2 = v.y;
                def.histop3 = v.z;
            }
            return def;
        }

        private ConfigurableJoint CreateBallAndSocketWithAMotor(Transform tr, Rigidbody body2)
        {
            if (tr == null)
                return null;

            HingeJoint _hingeMotor = tr.gameObject.GetComponent<HingeJoint>();
            if (_hingeMotor != null)
                DestroyImmediate(_hingeMotor);

            ConfigurableJoint motor = tr.gameObject.GetComponent<ConfigurableJoint>();
            if (motor == null)
            {
                motor = tr.gameObject.AddComponent<ConfigurableJoint>();
            }
            motor.connectedBody = body2;

            //set some default axes. Note that Rig class overrides these in AddBone
            motor.axis = body2.transform.right;
            motor.secondaryAxis = body2.transform.forward;

            motor.xMotion = motor.yMotion = motor.zMotion = ConfigurableJointMotion.Locked;
            motor.angularXMotion = motor.angularYMotion = motor.angularZMotion = ConfigurableJointMotion.Limited;

            SoftJointLimit _h0 = motor.highAngularXLimit;
            SoftJointLimit _l0 = motor.lowAngularXLimit;
            _l0.limit = this.lostop; motor.lowAngularXLimit = _l0;
            _h0.limit = this.histop; motor.highAngularXLimit = _h0;

            SoftJointLimit _l1 = motor.angularYLimit;
            _l1.limit = Mathf.Max(Mathf.Abs(this.lostop2), Mathf.Abs(this.histop2));
            motor.angularYLimit = _l1;

            SoftJointLimit _l2 = motor.angularZLimit;
            _l2.limit = Mathf.Max(Mathf.Abs(this.lostop3), Mathf.Abs(this.histop3));
            motor.angularZLimit = _l2;

            motor.rotationDriveMode = RotationDriveMode.Slerp;
            JointDrive _motorDrive = motor.slerpDrive;
            _motorDrive.positionSpring = kp;
            _motorDrive.positionDamper = kd;
            _motorDrive.maximumForce = maxForce;
            motor.slerpDrive = _motorDrive;

            motor.projectionMode = JointProjectionMode.PositionAndRotation;
            motor.enablePreprocessing = false;
            motor.configuredInWorldSpace = false;
            return motor;
        }

        #endregion

        public static JointDefinition CreateFixed()
        {
            JointDefinition def = new JointDefinition();
            def.type = Type.Fixed;
            return def;
        }

        public Type type;
        public float lostop;
        public float histop;
        public float lostop2;
        public float histop2;
        public float lostop3;
        public float histop3;

        public Vector3? axis;

        float kp = HumanoidMecanimRig.maxJointSpring; //spring constant // 10000
        float kd = HumanoidMecanimRig.jointDampen; //spring damping // 50
        float maxForce = HumanoidMecanimRig.maxJointForceLimit; // joint max force // 25000f
    }

    public class HumanoidBodyState
    {
        const int boneCount = 15;
        const int nAngle = 30;

        private void InitVariables()
        {
            pos = new Vector3[boneCount];
            rot = new Quaternion[boneCount];
            vel = new Vector3[boneCount];
            aVel = new Vector3[boneCount];

            currentSetAngles = new float[nAngle];
        }

        public HumanoidBodyState()
        {
            InitVariables();
        }

        public HumanoidBodyState(HumanoidBodyState _c)
        {
            InitVariables();
            Copy(_c);
        }

        public void Copy(HumanoidBodyState _c)
        {
            for (int i = 0; i < boneCount; i++)
            {
                pos[i] = _c.pos[i];
                rot[i] = _c.rot[i];
                vel[i] = _c.vel[i];
                aVel[i] = _c.aVel[i];
            }
            return;
        }

        public void SaveBoneState(List<BoneData> _cData)
        {
            for (int i = 0; i < _cData.Count; i++)
            {
                pos[i] = _cData[i].controlTransform.localPosition;
                rot[i] = _cData[i].controlTransform.localRotation;
                vel[i] = _cData[i].rb.velocity;
                aVel[i] = _cData[i].rb.angularVelocity;
            }
            return;
        }

        public void LoadBoneState(ref List<BoneData> _cData)
        {
            for (int i = 0; i < _cData.Count; i++)
            {
                _cData[i].controlTransform.localPosition = pos[i];
                _cData[i].controlTransform.localRotation = rot[i];
                
                _cData[i].rb.velocity = vel[i];
                _cData[i].rb.angularVelocity = aVel[i];
            }
        }

        public int GetBoneCount()
        {
            return boneCount;
        }

        public Quaternion GetBoneAngle(int b) { return rot[b]; }
        public Vector3 GetBonePos(int b) { return pos[b]; }
        public Vector3 Vel(int b) { return vel[b]; }
        public Vector3 AVel(int b) { return aVel[b]; }

        // these variables comes from bones
        Vector3[] pos;
        Quaternion[] rot;
        Vector3[] vel;
        Vector3[] aVel;

        public float[] currentSetAngles;
    };

    public class MemoryManager
    {
        public List<HumanoidBodyState> savedStates = new List<HumanoidBodyState>();
        public int GetNextFreeSlot()
        {
            savedStates.Add(new HumanoidBodyState());
            return savedStates.Count - 1;
        }
    };

    [Header("Joint Drive Settings")]
    public const float maxJointSpring = 100000f;
    public const float jointDampen = 5000f;
    public const float maxJointForceLimit = 25000f;
    [Header("Simulation parameters")]
    public float chestThicknessMul = 1;
    public float armsThicknessMul = 0.8f;
    public float neckThicknessMul = 1;
    //   public float maxTargetVel = Mathf.PI * 2.0f; //radians per seconds, applies to all motors. See driveMotorToTarget()
    public float totalMassKg = 70.0f;
    [Header("Joint limits")]
    public float hipSwingFwd = 110.0f;
    public float hipSwingBack = 20.0f;
    public float hipSwing = 45.0f;
    public float hipTwist = 45.0f;
    public float shoulderSwingFwd = 120.0f;
    public float shoulderSwingBack = 45.0f;
    public float shoulderSwing = 70.0f;
    public float shoulderTwistRange = 45.0f;
    public float spineSwingSideways = 20.0f;
    public float spineSwingForward = 40.0f;
    public float spineSwingBack = 10.0f;
    public float spineTwist = 45.0f;
    public float wristSwing = 15.0f;
    public float wristSwingOutwards = 70.0f;
    public float wristSwingInwards = 15.0f;
    public float wristTwistRange = 30.0f;
    public float ankleSwingRange = 30.0f;
    public float kneeSwingRange = 140.0f;
    [HideInInspector] public bool initialized = false;
    [HideInInspector] public Vector3 initialBiasPosition = new Vector3();
    [HideInInspector] public float[] minAngle, maxAngle;
    [HideInInspector] public MemoryManager mMemory = new MemoryManager();

    void Start()
    {
        initialBiasPosition = transform.position;
        initialize();
    }

    // only used for interface commands
    void Update()
    {
        
    }

    float[] cAngles = new float[30];
    float[] nAngles = new float[30];
    float[] cAngleRates = new float[30];
    private void FixedUpdate()
    {
        poseToMotorAngles(ref nAngles);
        for (int i = 0; i < 30; i++)
        {
            cAngleRates[i] = (nAngles[i] - cAngles[i]) / Time.fixedDeltaTime;
            cAngles[i] = nAngles[i];
        }
    }

    public void setCharacterPosition(Vector3 _p)
    {
        bones[0].controlTransform.localPosition = _p;
    }

    public void setCharacterRotation(Quaternion _r)
    {
        bones[0].controlTransform.localRotation = _r;
    }

    public List<float> readAnglePos()
    {
        List<float> out_list = new List<float>();

        FileStream posefile = new FileStream("climberAnglePos.txt", FileMode.Open);
        StreamReader reader = new StreamReader(posefile);
        for (int i = 0; i < numControlDOFs(); i++)
        {
            string _str = reader.ReadLine();
            out_list.Add(float.Parse(_str));
        }
        reader.Close();
        posefile.Close();
        return out_list;
    }

    public void initialize()
    {
        if (anim == null)
            initialized = false;
        if (initialized)
            return;
        
        anim = GetComponent<Animator>();
        if (anim != null)
        {
            Debug.Log("MotionOptimizer: The target character rig type appears to be a Mecanim Biped, building...");
            if (!anim.isHuman)
                Debug.LogException(new System.Exception("The target character is not a human. Check import settings."));
            targetCharacter = transform;
            buildMecanimBipedRig(false);
            setJointToBoneInfo();
            //            printRig();
            //            setKinematic(false);
            //            initJointMotors(Time.fixedDeltaTime, 10000, 1000, 1.0f, motorFmax);
            //            setFMaxForAllMotors(motorFmax);
        }
        else
        {
            Debug.LogException(new System.Exception("The target character is missing an Animator component!"));
        }

        initialized = true;
	}

    public void freezeBody(bool isFreezed)
    {
        if (anim == null)
            return;

        BoneData hip = mecanimBoneToBoneData[(int)HumanBodyBones.Hips];
        //BoneData spine = mecanimBoneToBoneData[(int)HumanBodyBones.Chest];

        if (isFreezed)
        {
            Rigidbody hipRB = hip.rb;
            hipRB.constraints = RigidbodyConstraints.FreezeAll;

            //Rigidbody spineRB = spine.rb;
            //spineRB.constraints = RigidbodyConstraints.FreezeAll;
        }
        else
        {
            Rigidbody hipRB = hip.rb;
            hipRB.constraints = RigidbodyConstraints.None;

            //Rigidbody spineRB = spine.rb;
            //spineRB.constraints = RigidbodyConstraints.None;
        }
    }

    public int numControlDOFs()
    {
        return numFreedoms;
    }

    public float[] angleMinLimits()
    {
        return minAngle;
    }

    public float[] angleMaxLimits()
    {
        return maxAngle;
    }

    public void getMotorAngleRates(ref float[] motorAngleRates)
    {
        if (motorAngleRates.Length < this.numControlDOFs())
            motorAngleRates = new float[this.numControlDOFs()];

        int curAngleIdx = 0;
        for (int i = 0; i < bones.Count; i++)
        {
            BoneData bi = bones[i];
            int dof = bi.numFreedoms;

            switch (dof)
            {
                case 3:
                    motorAngleRates[curAngleIdx] = cAngleRates[curAngleIdx];
                    motorAngleRates[curAngleIdx + 1] = cAngleRates[curAngleIdx + 1];
                    motorAngleRates[curAngleIdx + 2] = cAngleRates[curAngleIdx + 2];
                    break;
                case 1:
                    motorAngleRates[curAngleIdx] = cAngleRates[curAngleIdx];
                    break;
            }

            curAngleIdx += dof;
        }
    }

    public void poseToMotorAngles(ref float[] motorAngles)
    {
        if (motorAngles.Length < this.numControlDOFs())
            motorAngles = new float[this.numControlDOFs()];

        int curAngleIdx = 0;
        for (int i = 0; i < bones.Count; i++)
        {
            BoneData bi = bones[i];
            int dof = bi.numFreedoms;

            switch (dof)
            {
                case 3:
                    motorAngles[curAngleIdx] = bi.GetCurrentAngleAxis(0);
                    motorAngles[curAngleIdx + 1] = bi.GetCurrentAngleAxis(1);
                    motorAngles[curAngleIdx + 2] = bi.GetCurrentAngleAxis(2);
                    break;
                case 1:
                    motorAngles[curAngleIdx] = bi.GetCurrentAngleAxis(0);
                    break;
            }

            curAngleIdx += dof;
        }
    }

    public void getCurrentAplliedTorques(ref float[] motorTorques)
    {
        int curAngleIdx = 0;
        for (int i = 0; i < bones.Count; i++)
        {
            BoneData bi = bones[i];
            int dof = bi.numFreedoms;

            switch (dof)
            {
                case 3:
                    motorTorques[curAngleIdx] = bi.GetTorqueJointAxis(0);
                    motorTorques[curAngleIdx + 1] = bi.GetTorqueJointAxis(1);
                    motorTorques[curAngleIdx + 2] = bi.GetTorqueJointAxis(2);
                    break;
                case 1:
                    motorTorques[curAngleIdx] = bi.GetTorqueJointAxis(0);
                    break;
            }

            curAngleIdx += dof;
        }
    }

    [HideInInspector] public float[] cTargetAngles = new float[30];
    public void driveToPose(float[] angles, int sIndex = 0)//, float timeStep)
    {
        int curAngleIdx = sIndex;
        for (int i = 0; i < bones.Count; i++)
        {
            BoneData bi = bones[i];

            Rigidbody rb = bi.rb;
            if (rb.IsSleeping()) rb.WakeUp();

            int dof = bi.numFreedoms;

            for (int idx = 0; idx < dof; idx++)
            {
                cTargetAngles[curAngleIdx + idx] = angles[curAngleIdx + idx];
                if (angles[curAngleIdx + idx] > maxAngle[curAngleIdx + idx])
                    angles[curAngleIdx + idx] = maxAngle[curAngleIdx + idx];
                if (angles[curAngleIdx + idx] < minAngle[curAngleIdx + idx])
                    angles[curAngleIdx + idx] = minAngle[curAngleIdx + idx];
            }

            switch (dof)
            {
                case 3:
                    bi.SetAMotorAngles(angles[curAngleIdx], angles[curAngleIdx + 1], angles[curAngleIdx + 2]);
                    bi.currentXNormalizedRot = Mathf.InverseLerp(minAngle[curAngleIdx], maxAngle[curAngleIdx], angles[curAngleIdx]);
                    bi.currentYNormalizedRot = Mathf.InverseLerp(minAngle[curAngleIdx + 1], maxAngle[curAngleIdx + 1], angles[curAngleIdx + 1]);
                    bi.currentZNormalizedRot = Mathf.InverseLerp(minAngle[curAngleIdx + 2], maxAngle[curAngleIdx + 2], angles[curAngleIdx + 2]);
                    break;
                case 1:
                    bi.SetHingeAngle(angles[curAngleIdx]);
                    bi.currentXNormalizedRot = Mathf.InverseLerp(minAngle[curAngleIdx], maxAngle[curAngleIdx], angles[curAngleIdx]);
                    bi.currentYNormalizedRot = 0.5f;
                    bi.currentZNormalizedRot = 0.5f;
                    break;
            }
            curAngleIdx += dof;
        }
        return;
    }

    public Vector3 COM()
    {
        float mass = 0.0f;
        Vector3 comAccum = Vector3.zero;
        foreach (BoneData bd in bones)
        {
            mass += bd.rb.mass;
            comAccum += bd.rb.mass * (bd.rb.position - initialBiasPosition);
        }
        comAccum *= (1.0f / mass);
        return comAccum;
    }

    public Vector3 COMVel()
    {
        float mass = 0.0f;
        Vector3 comAccum = Vector3.zero;
        foreach (BoneData bd in bones)
        {
            mass += bd.rb.mass;
            comAccum += bd.rb.mass * bd.rb.velocity;
        }
        comAccum *= (1.0f / mass);
        return comAccum;
    }

    public Vector3 pos(HumanBodyBones bone)
    {
        return mecanimBoneToBoneData[(int)bone].rb.position - initialBiasPosition;
    }

    public Vector3 getWorldBonePos(HumanBodyBones bone)
    {
        return mecanimBoneToBoneData[(int)bone].rb.position;
    }

    public GameObject getBoneGameObject(HumanBodyBones bone)
    {
        return mecanimBoneToBoneData[(int)bone].controlObject;
    }

    public Transform getBoneTransform(HumanBodyBones bone)
    {
        if (!extraHeadJoint && bone == HumanBodyBones.Head)
            bone = HumanBodyBones.Neck;
        return mecanimBoneToBoneData[(int)bone].controlTransform;
    }

    public Vector3 vel(HumanBodyBones bone)
    {
        return mecanimBoneToBoneData[(int)bone].rb.velocity;
    }

    public Vector3 avel(HumanBodyBones bone)
    {
        return mecanimBoneToBoneData[(int)bone].rb.angularVelocity;
    }

    public Quaternion q(HumanBodyBones bone)
    {
        return mecanimBoneToBoneData[(int)bone].rb.rotation;
    }

    public int boneIdx(HumanBodyBones bone)
    {
        return mecanimBoneToBoneData[(int)bone].index;
    }

    public float boneSize(HumanBodyBones bone)
    {
        return mecanimBoneToBoneData[(int)bone].boneSize;
    }

    public Quaternion getBoneAngle(int boneID)
    {
        return q(bones[boneID].mecanimBone);
    }

    public Vector3 getBonePos(int boneID)
    {
        return pos(bones[boneID].mecanimBone);
    }

    public Vector3 getWorldBonePos(int boneID)
    {
        return bones[boneID].rb.position;
    }
    
    public Vector3 getBoneAVel(int boneID)
    {
        return avel(bones[boneID].mecanimBone);
    }

    public Vector3 getBoneVel(int boneID)
    {
        return vel(bones[boneID].mecanimBone);
    }

    public void GetCurrentJointForces()
    {
        for (int i = 0; i < bones.Count; i++)
        {
            bones[i].currentJointForce = bones[i].GetForceJoint();
            bones[i].currentJointForceSqrMag = bones[i].GetForceJoint().magnitude;
            bones[i].currentJointTorque = bones[i].GetTorqueJoint();
            bones[i].currentJointTorqueSqrMag = bones[i].GetTorqueJoint().magnitude;
        }
    }

    public int getBonesCount()
    {
        return bones.Count;
    }

    // given a memory index it loads the info from memory to context
    public bool loadState(int _slotIndex)
    {
        if (mMemory == null)
            return false;
        if (mMemory.savedStates == null)
            return false;
        if (_slotIndex >= 0 && _slotIndex < mMemory.savedStates.Count)
        {
            mMemory.savedStates[_slotIndex].LoadBoneState(ref bones);

            driveToPose(mMemory.savedStates[_slotIndex].currentSetAngles);

            return true;
        }
        return false;
    }

    // given an initilized state it loads that state into context
    public bool loadState(ref HumanoidBodyState nState)
    {
        nState.LoadBoneState(ref bones);
        
        driveToPose(nState.currentSetAngles);
        
        return true;
    }

    // given a free memory index, this function saves current state of the context into the memory
    public bool saveState(int _freeSlotIdx)
    {
        if (mMemory == null)
            return false;
        if (mMemory.savedStates == null)
            return false;

        if (_freeSlotIdx >= 0 && _freeSlotIdx < mMemory.savedStates.Count)
        {
            mMemory.savedStates[_freeSlotIdx].SaveBoneState(bones);

            for (int i = 0; i < numFreedoms; i++)
            {
                mMemory.savedStates[_freeSlotIdx].currentSetAngles[i] = cTargetAngles[i];
            }
            
            return true; // successful saving
        }

        return false;
    }

    public int getNumBones()
    {
        return bones.Count;
    }

    public Dictionary<int, BoneData> jointToBone = new Dictionary<int, BoneData>();

    public BoneData getBodyFromJointIndex(int j)
    {
        return jointToBone[j];
    }

    public BoneData getBodyFromBoneIdx(int b)
    {
        return bones[b];
    }
    
    public float getMotorAppliedSqTorque(int j)
    {
        return (jointToBone[j].GetTorqueJoint().sqrMagnitude) / (float)(jointToBone[j].numFreedoms);//+ jointToBone[j].GetForceJoint().sqrMagnitude
    }

    public float getMotorAppliedSqForce(int b)
    {
        return bones[b].GetForceJoint().sqrMagnitude;
    }

    public float getControlEffort()
    {
        float effort = 0;
        for (int i = 0; i < bones.Count; i++)
        {
            effort += bones[i].GetTorqueJoint().magnitude;
        }
        return effort;
    }

    ///////////////////////////////////////////////////////// private functions

    void buildMecanimBipedRig(bool constrainTo2d = false)
    {
        //Compute character forward and right vectors, assuming character upright in t-pose
        //These will be used to figure out which bone local vectors (right, up, forward) are mapped to which motor axes.
        //Note that we assume that the rig is built so that one vector points along the bone and one points forward.
        tposeRight = anim.GetBoneTransform(HumanBodyBones.RightFoot).position - anim.GetBoneTransform(HumanBodyBones.LeftFoot).position;
        tposeRight.y = 0;
        tposeRight.Normalize();
        //Left hand rule: cross-product lhs is thumb, rhs is index finger, result is middle finger
        tposeForward = Vector3.Cross(tposeRight, Vector3.up);
        //            Debug.Log("Character forward " + tposeForward + ", right " + tposeRight);

        //For ball joints (others than knee, angle, elbow), 
        //axis 0 points forward, positive angles rotate counterclockwise
        //axis 1 points along the bone
        //axis 2 axis points right in t-pose for legs, up for arms
        bool addSpineJoint = false;
        bool addHeadJoint = extraHeadJoint;
        this.constrainTo2d = constrainTo2d;

        rootUp = new Vector3(0, 1, 0);
        //spine to head
        Transform hip = anim.GetBoneTransform(HumanBodyBones.Hips);
        Transform spine = anim.GetBoneTransform(HumanBodyBones.Spine);
        Transform chest = anim.GetBoneTransform(HumanBodyBones.Chest);
        Transform neck = anim.GetBoneTransform(HumanBodyBones.Neck);
        // Transform head = anim.GetBoneTransform(HumanBodyBones.Head);

        AddBone(hip, addSpineJoint ? spine : chest, null, JointDefinition.CreateFixed());
        //fix pelvis to xy plane if in 2d mode
        if (constrainTo2d && createPhysicsObjects)
        {
            //int jointId = UnityOde.odeJointCreatePlane2D();
            //UnityOde.odeJointAttach(jointId, bones[0].rb.BodyId, 0);
        }
        //    (bones[bones.Count - 1].c as OdeGeomCapsule).SetDensity(2.0f*humanDensity); //double the density to account for the torso volume not represented by the capsule

        if (addSpineJoint)
        {
            var spineJoint = constrainTo2d ?
                JointDefinition.CreateHinge(-45.0f, 45.0f) :
                JointDefinition.CreateBallAndSocket(new Vector3(-spineSwingBack,- spineSwingSideways, -spineTwist),
                    new Vector3(spineSwingForward, spineSwingSideways, spineTwist));
            AddBone(spine, chest, hip, spineJoint);
        }

        //chest to head
        var chestJointDef = constrainTo2d ?
            JointDefinition.CreateHinge(-45.0f, 45.0f) :
                JointDefinition.CreateBallAndSocket(new Vector3(-spineSwingBack, - spineSwingSideways, -spineTwist),
                    new Vector3(spineSwingForward, spineSwingSideways, spineTwist));
        AddBone(chest, neck, addSpineJoint ? spine : hip, chestJointDef);
        //Debug.Log("Chest index: " + (bones.Count - 1));

        var neckJoint = constrainTo2d ?
            JointDefinition.CreateHinge(-45, 45) :
                JointDefinition.CreateBallAndSocket(new Vector3(-spineSwingBack * 0.5f, - spineSwingSideways * 0.5f, -spineTwist * 0.5f),
                    new Vector3(spineSwingForward * 0.5f, spineSwingSideways * 0.5f, spineTwist * 0.5f));
        AddBone(HumanBodyBones.Neck, addHeadJoint ? HumanBodyBones.Head : HumanBodyBones.Jaw, HumanBodyBones.Chest, neckJoint);

        if (addHeadJoint)
        {
            var headJoint = constrainTo2d ?
                JointDefinition.CreateHinge(-45, 45) :
                    JointDefinition.CreateBallAndSocket(new Vector3(-spineSwingBack, - spineSwingSideways, -spineTwist),
                        new Vector3(spineSwingForward, spineSwingSideways, spineTwist));
            AddBone(HumanBodyBones.Head, HumanBodyBones.Jaw, HumanBodyBones.Neck, headJoint);
        }

        //left leg
        var leftHipJointDef = constrainTo2d ?
            JointDefinition.CreateHinge(-90.0f, 90.0f) :
            JointDefinition.CreateBallAndSocket(new Vector3(-hipSwingBack, -hipSwing, -hipTwist),
                new Vector3(hipSwingFwd, hipSwing, hipTwist));
        AddBone(HumanBodyBones.LeftUpperLeg, HumanBodyBones.LeftLowerLeg, HumanBodyBones.Hips, leftHipJointDef);
        AddBone(HumanBodyBones.LeftLowerLeg, HumanBodyBones.LeftFoot, HumanBodyBones.LeftUpperLeg, JointDefinition.CreateHinge(10, kneeSwingRange));

        var ankleTwist = 10.0f;
        var ankleRotate = 30.0f;
        var ankleDef = hingeAnkle ? JointDefinition.CreateHinge(-ankleSwingRange, ankleSwingRange)
            : JointDefinition.CreateBallAndSocket(new Vector3(-ankleTwist, -ankleSwingRange, -ankleRotate),
                new Vector3(ankleTwist, ankleSwingRange, ankleRotate));
        AddBone(HumanBodyBones.LeftFoot, HumanBodyBones.LeftToes, HumanBodyBones.LeftLowerLeg, ankleDef);

        //right leg
        var rightHipJointDef = constrainTo2d ?
            JointDefinition.CreateHinge(-90.0f, 90.0f) :
            JointDefinition.CreateBallAndSocket(new Vector3(-hipSwingBack, -hipSwing, -hipTwist),
                new Vector3(hipSwingFwd, hipSwing, hipTwist));
        AddBone(HumanBodyBones.RightUpperLeg, HumanBodyBones.RightLowerLeg, HumanBodyBones.Hips, rightHipJointDef);
        AddBone(HumanBodyBones.RightLowerLeg, HumanBodyBones.RightFoot, HumanBodyBones.RightUpperLeg, JointDefinition.CreateHinge(10, kneeSwingRange));
        AddBone(HumanBodyBones.RightFoot, HumanBodyBones.RightToes, HumanBodyBones.RightLowerLeg, ankleDef);


        //left arm
        var leftShoulderJointDef = constrainTo2d ?
            JointDefinition.CreateHinge(-90.0f, 90.0f) :
            JointDefinition.CreateBallAndSocket(new Vector3(-shoulderSwingBack, -shoulderSwing, -shoulderTwistRange),
                new Vector3(shoulderSwingFwd, shoulderSwing, shoulderTwistRange));
        AddBone(HumanBodyBones.LeftUpperArm, HumanBodyBones.LeftLowerArm, HumanBodyBones.Chest, leftShoulderJointDef);

        AddBone(HumanBodyBones.LeftLowerArm, HumanBodyBones.LeftHand, HumanBodyBones.LeftUpperArm, JointDefinition.CreateHinge(-140, 0.0f));

        var leftWristJointDef = constrainTo2d ?
            JointDefinition.CreateHinge(-30.0f, 30.0f) :
            JointDefinition.CreateBallAndSocket(new Vector3(-wristSwingOutwards, -wristTwistRange, -wristSwing),
                new Vector3(wristSwingInwards, wristTwistRange, wristSwing));
        //Vector3 elbowToWrist = anim.GetBoneTransform(HumanBodyBones.LeftHand).position - anim.GetBoneTransform(HumanBodyBones.LeftLowerArm).position;
        Vector3 fingerTipEstimate = anim.GetBoneTransform(HumanBodyBones.LeftMiddleDistal).position;
        AddBone(HumanBodyBones.LeftHand, fingerTipEstimate, HumanBodyBones.LeftLowerArm, leftWristJointDef);


        //right arm
        var rightShoulderJointDef = constrainTo2d ?
            JointDefinition.CreateHinge(-90.0f, 90.0f) :
            JointDefinition.CreateBallAndSocket(new Vector3(-shoulderSwingBack, -shoulderSwing, -shoulderTwistRange),
                new Vector3(shoulderSwingFwd, shoulderSwing, shoulderTwistRange));
        AddBone(HumanBodyBones.RightUpperArm, HumanBodyBones.RightLowerArm, HumanBodyBones.Chest, rightShoulderJointDef);

        AddBone(HumanBodyBones.RightLowerArm, HumanBodyBones.RightHand, HumanBodyBones.RightUpperArm, JointDefinition.CreateHinge(-140.0f, 0.0f));

        var rightWristJointDef = constrainTo2d ?
            JointDefinition.CreateHinge(-30.0f, 30.0f) :
            JointDefinition.CreateBallAndSocket(new Vector3(-wristSwingInwards, -wristTwistRange, -wristSwing),
                new Vector3(wristSwingOutwards, wristTwistRange, wristSwing));
        //elbowToWrist = anim.GetBoneTransform(HumanBodyBones.RightHand).position - anim.GetBoneTransform(HumanBodyBones.RightLowerArm).position;
        fingerTipEstimate = anim.GetBoneTransform(HumanBodyBones.RightMiddleDistal).position;
        AddBone(HumanBodyBones.RightHand, fingerTipEstimate, HumanBodyBones.RightLowerArm, rightWristJointDef);


        //scale total mass to 70kg
        if (createPhysicsObjects)
        {
            float mass = 0;
            foreach (BoneData data in bones)
            {
                data.rb.SetDensity(humanDensity);
                mass += data.rb.mass;
            }
            float scaleFactor = totalMassKg / mass;
 //           Debug.LogWarning("Rig physics mass " + mass + ", scaling mass by " + scaleFactor);
            foreach (BoneData data in bones)
            {
                data.rb.mass = data.rb.mass * scaleFactor;
                data.cachedMass = data.rb.mass;
            }
        }
        //finalize
        //        if (createPhysicsObjects)
        //            disableBoneToBoneCollisions();
        numFreedoms = 0;
        foreach (BoneData bd in bones)
        {
            bd.firstAngleIndex = numFreedoms;
            numFreedoms += bd.numFreedoms;
        }
        updateAngleLimits();
        initBonelessTransforms();
        return;
    }

    void setJointToBoneInfo()
    {
        int curAngleIdx = 0;
        for (int i = 0; i < bones.Count; i++)
        {
            BoneData bi = bones[i];

            Rigidbody rb = bi.rb;
            if (rb.IsSleeping()) rb.WakeUp();

            int dof = bi.numFreedoms;

            for (int idx = 0; idx < dof; idx++)
            {
                jointToBone.Add(curAngleIdx, bi); curAngleIdx++;
            }
        }

        return;
    }

    Vector3 rootUp=Vector3.up;
	Transform targetCharacter;
    Animator anim=null;
    bool hingeAnkle = true;
//	//	float feetY=0;
    bool constrainTo2d;
	bool createPhysicsObjects=true;  //set before calling buildMecanimBipedRig
    Dictionary<Transform, BoneData> startTransformToBoneData = new Dictionary<Transform, BoneData>();
    BoneData [] mecanimBoneToBoneData = new BoneData[(int)HumanBodyBones.LastBone];
    Dictionary<int, BoneData> rigidBodyIdToBoneData = new Dictionary<int, BoneData>();
    Dictionary<Transform, BonelessTransformData> bonelessTransformDefaults = new Dictionary<Transform, BonelessTransformData>();
    const float humanDensity = 1000.0f; //kg/m3
    //character orientation vectors, computed in buildMecanimBipedRig() assuming character in t-pose
    Vector3 tposeForward, tposeRight;
    bool extraHeadJoint = false;
    
    public List<BoneData> bones = new List<BoneData>();
    int numFreedoms = 0;

    static string GetRelativePath(Transform current, Transform relativeTo)
    {
        if (current.parent == relativeTo)
            return current.name;
        if (current.parent == null)
            return "/" + current.name;
        return GetRelativePath(current.parent, relativeTo) + "/" + current.name;
    }
    Transform findRecursive(Transform parent, string name)
    {
        Transform result = parent.Find(name);
        if (result != null)
            return result;
        foreach (Transform t in parent)
        {
            Transform child = findRecursive(t, name);
            if (child != null)
                return child;
        }
        return null;
    }
    Transform findRecursive(string name)
    {
        return findRecursive(targetCharacter, name);
    }
        
    HumanBodyBones boneTransformToHumanBodyBone(Transform boneTransform)
    {
        foreach (var b in System.Enum.GetValues(typeof(HumanBodyBones)))
        {
            if (anim.GetBoneTransform((HumanBodyBones)b) == boneTransform)
            {
                return (HumanBodyBones)b;
            }
        }
        Debug.LogException(new System.Exception("Could not map bone transform " + boneTransform.name + " to a mecanim HumanBodyBones enum!"));
        return HumanBodyBones.LastBone;
    }

    Vector3[] basisVectors = new Vector3[] { Vector3.right, Vector3.up, Vector3.forward, -Vector3.right, -Vector3.up, -Vector3.forward };

    int GetClosestLocalAxisIdx(Transform transform, Vector3 target)
    {
        float maxProj = -10;
        int result = -1;
        for (int i = 0; i < 6; i++)
        {
            Vector3 rotated = transform.TransformDirection(basisVectors[i]);
            float proj = Vector3.Dot(rotated, target);
            if (proj > maxProj)
            {
                result = i;
                maxProj = proj;
            }
        }
        return result;
    }

    int GetBipedHingeLocalAxisIdx(HumanBodyBones boneName)
    {
        BoneData data = startTransformToBoneData[anim.GetBoneTransform(boneName)];
        Vector3 desiredHingeVector;
        //All biped bones with hinge joints except feet seem to follow the same logic, assuming that one of the local axes points forward and one along the bone.
        //Feet bones are different since they are the only ones pointing forward in the t-pose
        if (boneName == HumanBodyBones.LeftFoot || boneName == HumanBodyBones.RightFoot)
        {

            desiredHingeVector = Vector3.Cross(Vector3.up, data.boneVector.normalized);
        }
        else
        {
            desiredHingeVector = Vector3.Cross(tposeForward, data.boneVector.normalized);
        }
        return GetClosestLocalAxisIdx(data.controlTransform, desiredHingeVector);
    }
    
    void AddBone(Transform startTransform, Vector3 boneEndPos, Transform parentTransform, JointDefinition def)
    {
        float parentScale = HumanoidClimberAgent.localAgentScaling;// gameObject.transform.localScale[0];

        bool isFoot = (startTransform == anim.GetBoneTransform(HumanBodyBones.RightFoot)
            || startTransform == anim.GetBoneTransform(HumanBodyBones.LeftFoot));
        bool isHead=startTransform == anim.GetBoneTransform(HumanBodyBones.Neck);
        bool isHand = (startTransform == anim.GetBoneTransform(HumanBodyBones.RightHand)
            || startTransform == anim.GetBoneTransform(HumanBodyBones.LeftHand));
        //The physics bone controls the startTransform 
        GameObject controlObject = startTransform.gameObject;

        //Some bookkeeping. All relevant data is stored in a BoneData instance, and 
        //the startTransformToBoneData dictionary can be used to look up BoneData for a start transform.
        //Example: for a forearm bone, startTransform's origin is at elbow, endTransform's origin at wrist, and parentTransform's origin at shoulder.
        BoneData data = new BoneData();
        startTransformToBoneData.Add(startTransform, data);

        data.mecanimBone = boneTransformToHumanBodyBone(startTransform);
        mecanimBoneToBoneData[(int)data.mecanimBone]=data;

        float thickness = 0.1f;

        //change this to make the decisions simply using the mecanim bone names and e.g., Transform.isChildOf
        switch (data.mecanimBone)
        {
            case HumanBodyBones.Neck:
                thickness *= neckThicknessMul;
                break;
            case HumanBodyBones.Chest:
            case HumanBodyBones.Hips:
                thickness *= 2f;
                break;
            case HumanBodyBones.Spine:
                thickness *= chestThicknessMul;
                break;
            case HumanBodyBones.LeftHand:
            case HumanBodyBones.LeftLowerArm:
            case HumanBodyBones.LeftUpperArm:
            case HumanBodyBones.RightHand:
            case HumanBodyBones.RightLowerArm:
            case HumanBodyBones.RightUpperArm:
                thickness *= armsThicknessMul;
                break;
        }

        //The bone will be represented using a physics capsule. We'll now figure out the
        //capsule's dimensions and orientation.
        Vector3 boneStartPos = startTransform.position;
        if (isFoot)
        {
           // thickness *= 0.7f;
            //hack: in most biped rigs, the foot (ankle) transform is higher than the toe, but we want a horizontal foot bone in the T-pose
            boneEndPos.y += thickness * 0.5f;
            boneEndPos.y += 0.01f;
            boneStartPos.y = boneEndPos.y + 0.025f; // debugging the end point of feet was rotating downward!, adding 0.025f to fix it
            boneStartPos += tposeForward * 0.02f;
            boneEndPos += tposeForward * 0.02f;
            data.FmaxScale = 1.0f;
        }
        if (isHand)
        {
            data.FmaxScale = 0.5f;
        }
        if (isHead)
            boneEndPos+=(boneEndPos-boneStartPos) * 0.5f;

        Vector3 boneVector = (boneEndPos - boneStartPos);

        //link to parent bone
        Rigidbody parentBody = null;
        if (parentTransform != null)
        {
            data.parent = startTransformToBoneData[parentTransform];
            parentBody = data.parent.rb;
            if (parentBody == null && createPhysicsObjects)
                Debug.LogError("Could not find parent bone for " + controlObject.name);
        }
        else
        {
            data.parent = null;
        }

        //fill in rest of the BoneData
        data.rb = null;
        data.c = null;
        data.joint = null;
        data.relativeControlPath = GetRelativePath(startTransform, targetCharacter);
        data.controlTransform = startTransform;
        data.parentTransform = parentTransform;
        data.controlObject = controlObject;
        data.index = bones.Count;
        data.name = startTransform.name;

        data.boneVector = boneVector;
        data.jointDef = def;

        // add climbing info data
        data.groundContact = data.controlTransform.GetComponent<GroundContact>();
        if (!data.groundContact)
        {
            data.groundContact = data.controlTransform.gameObject.AddComponent<GroundContact>();
        }
        if (!isFoot)
        {
            data.groundContact.groundContactPenalty = -1f;
            data.groundContact.penalizeGroundContact = true;
            data.groundContact.agentDoneOnGroundContact = true;
        }
        data.groundContact.agent = gameObject.GetComponent<Agent>();

        data.wallContact = data.controlTransform.GetComponent<WallContact>();
        if (!data.wallContact)
        {
            data.wallContact = data.controlTransform.gameObject.AddComponent<WallContact>();
        }
        data.wallContact.agent = gameObject.GetComponent<Agent>();

        data.holdContact = data.controlTransform.GetComponent<HoldContact>();
        if (!data.holdContact)
        {
            data.holdContact = data.controlTransform.gameObject.AddComponent<HoldContact>();
        }
        data.holdContact.agent = gameObject.GetComponent<Agent>();

        // Add & setup the target contact script
        data.targetContact = data.controlTransform.GetComponent<TargetContact>();
        if (!data.targetContact)
        {
            data.targetContact = data.controlTransform.gameObject.AddComponent<TargetContact>();
        }

        bones.Add(data);

        if (!createPhysicsObjects)
            return;

        //create and init a capsule geom
        float len = Mathf.Max(thickness, boneVector.magnitude);//-thickness*0.5f);
        if (isFoot)
        {
            len *= 1.5f;// 2.8f//3.4f; //hack the toe transform is actually in the middle of the foot -> elongate
        }
        data.boneSize = len;

        BoxCollider _bc = controlObject.GetComponent<BoxCollider>();
        if (_bc != null)
            DestroyImmediate(_bc);
        CapsuleCollider cc = controlObject.GetComponent<CapsuleCollider>();
        if (cc ==null)
            cc = controlObject.AddComponent<CapsuleCollider>();
        if (isHand)
        {
            cc.center = cc.center / 2.0f;
        }
        cc.radius = thickness * 0.5f;
        cc.height = len / parentScale;
        data.c = cc;

        // disable self-collisions except for legs where they're crucial for plausible mmovement
        if (data.parentTransform != null)
        {
            Physics.IgnoreCollision(data.parent.c, data.c);
        }

        data.thickness = thickness;

        // TODO: set material properties
        /*c.material=new PhysicMaterial();
        c.material.staticFriction=0.5f;
        c.material.dynamicFriction=0.5f;
        c.material.bounciness=0;
        c.material.bounceCombine=PhysicMaterialCombine.Minimum;
        c.material.frictionCombine=PhysicMaterialCombine.Average;*/
        //bone.renderer.enabled=false;
        //bone.transform.localRotation=Quaternion.FromToRotation(bone.transform.forward, boneVector.normalized);
        //bone.transform.rotation.SetLookRotation(boneVector.normalized, Vector3.up);
        //bone.transform.parent=startTransform;

        //create and init the rigid body
        Rigidbody rb = controlObject.GetComponent<Rigidbody>();
        if (rb == null)
            rb = controlObject.AddComponent<Rigidbody>();
        rigidBodyIdToBoneData.Add(rb.GetInstanceID(), data);
//          rb.position = boneStartPos + 0.5f * boneVector;
//          rb.rotation = Quaternion.LookRotation(boneVector.normalized, Vector3.up);
//          rb.isKinematic = true;
        rb.SetDensity(humanDensity); //will recompute mass

        data.rb = rb;
        data.initialRotation = rb.rotation;

        if (parentTransform != null)
        {
            data.parent = startTransformToBoneData[parentTransform];

            //TODO: get rid of this, currently BackgroundMotionOptimizer.SetRigToSplineTime depends on this
            if (data.parent.controlTransform != startTransform.parent)
            {
                List<Transform> transformsToParent = new List<Transform>();
                List<Vector3> localPositionsToParent = new List<Vector3>();
                List<Quaternion> localRotationsToParent = new List<Quaternion>();

                for (Transform parent = startTransform.parent; parent != null && parent != data.parent.controlTransform; parent = parent.parent)
                {
                    transformsToParent.Add(parent);
                    localPositionsToParent.Add(parent.localPosition);
                    localRotationsToParent.Add(parent.localRotation);
                }

                if (transformsToParent.Count == 0)
                {
                    Debug.LogError("Hierarchy error");
                }

                data.transformHierarchyToParent = transformsToParent.ToArray();
                data.localPositionHierarchyToParent = localPositionsToParent.ToArray();
                data.localRotationHierarchyToParent = localRotationsToParent.ToArray();
            }
        }
        else
        {
            data.parent = null;
        }
//        Debug.Log(rb.name + " phys: " + rb.rotation + " unity: " + controlObject.transform.rotation);

        //Body and Geometry now initialized. Now we need to create the joint if there is a parent body
        if (parentBody == null)
        {
            data.numFreedoms = 0;
        }
        else
        {
 //           Debug.Log("Connecting bone " + controlObject.name + " to " + parentBody.name);
            if (rb.GetInstanceID() == parentBody.GetInstanceID())
            {
                Debug.LogError(controlObject.name + ": Trying to create hinge joint between two bodies with same IDs: " + rb.name + " and " + parentBody.name + "(" + rb.GetInstanceID() + ")");
                return;
            }

            //create the joint, using the parent bone as Ode body 1
            data.joint = def.Create(startTransform, parentBody);
            data.hinge = data.joint as HingeJoint;

            //create and init a joint
            if (data.hinge != null)
            {
                //For the hinges, we want the rotation axis that is perpendicular to character forward and the bone vector.
                //For thighs and upperarms pointing down, this is equal to the tposeRight
                //Note that we don't just assume, e.g., hinge axis = tposeRight, since sometimes the t-pose may have arms 45 degrees downwards
                //for easier skin weight adjustments
                int hingeAxis = GetBipedHingeLocalAxisIdx(data.mecanimBone);
                data.hinge.axis = startTransform.TransformDirection(basisVectors[hingeAxis]);  //map local axis to global axis
 //               Debug.Log("Bone " + data.name + ": mapped hinge axis to local axis " + hingeAxis + ", " + data.hinge.axis);
                data.numFreedoms = 1;
                data.motor = data.hinge; //hinges have their own motors
                                         //the hingeAxis member is used by Pose class quite heavily, and the implementation assumes
                                         //that the range is 0..2, i.e., the hinge axis would directly correspond to a local axis of the mecanim rig, which
                                         //we have later found to not be enough. Some rigs have flipped axes, which is why we now have the 6 possible indices used above.
 //               data.hingeAxis = hingeAxis % 3;
            }
            else if (data.joint as ConfigurableJoint != null)
            {
                data.numFreedoms = 3;
                ConfigurableJoint motor = startTransform.GetComponent<ConfigurableJoint>();
                data.motor = motor;
                data.amotor = motor;
                if (motor == null)
                    Debug.LogException(new System.Exception("Bone ball and socket joint missing a motor"));

                Vector3 firstVecAxis = Vector3.Cross(tposeForward, boneVector.normalized);
                Vector3 secVecAxis = tposeForward;
                if (isHand)
                {
                    firstVecAxis = tposeForward;
                    secVecAxis = Vector3.Cross(tposeForward, boneVector.normalized);
                    
                }
                
                int motorAxis0idx = GetClosestLocalAxisIdx(startTransform, firstVecAxis);
                motor.axis = startTransform.TransformDirection(basisVectors[motorAxis0idx]);

                int motorAxis2idx = GetClosestLocalAxisIdx(startTransform, secVecAxis);
                motor.secondaryAxis = startTransform.TransformDirection(basisVectors[motorAxis2idx]);
                
 //               Debug.Log("Bone " + data.name + ": mapped motor axes 0,2 to local axes " + motorAxis0idx + "," + motorAxis2idx);
            }
            else
                Debug.LogError("Unknown joint type!");
        }
    }
    void AddBone(string start, string end, string parentName, JointDefinition def)
    {
        Transform startTransform = findRecursive(start);
        Transform endTransform = findRecursive(end);
        Transform parentTransform = null;
        if (parentName != null)
            parentTransform = findRecursive(parentName);
        AddBone(startTransform, endTransform.position, parentTransform, def);
    }
	void AddBone(HumanBodyBones start, HumanBodyBones end, HumanBodyBones parentName, JointDefinition def)
    {
        Transform endTransform = anim.GetBoneTransform(end);
        if (endTransform == null)
            Debug.LogError("Bone end transform null!");
        AddBone(start, endTransform.position, parentName, def);
    }
    void AddBone(HumanBodyBones start, Vector3 boneEndPos, HumanBodyBones parentName, JointDefinition def)
    {
        Transform startTransform = anim.GetBoneTransform(start);
        if (startTransform == null)
            Debug.LogError("Bone start transform null!");
        Transform parentTransform = null;
        if (parentName != start)
        {
            parentTransform = anim.GetBoneTransform(parentName);
            if (parentTransform == null)
                Debug.LogError("Bone parent transform null!");
        }
        AddBone(startTransform, boneEndPos, parentTransform, def);
    }
    void AddBone(Transform startTransform, Transform endTransform, Transform parentTransform, JointDefinition def)
    {
        AddBone(startTransform, endTransform.position, parentTransform, def);
    }
    
    void initBonelessTransforms()
    {
//        Debug.Log("Init boneless");
        bonelessTransformDefaults = getBonelessTransformStates();
    }
        
    void updateAngleLimits()
    {
        minAngle = new float[numControlDOFs()];
        maxAngle = new float[numControlDOFs()];
        int paramIndex = 0;
        for (int i = 0; i < bones.Count; i++)
        {
            BoneData boneData = bones[i];
            if (boneData.numFreedoms > 0)
            {
                JointDefinition def = boneData.jointDef;
                minAngle[paramIndex] = def.lostop;
                maxAngle[paramIndex] = def.histop;
                paramIndex++;
            }
            if (boneData.numFreedoms > 1)
            {
                JointDefinition def = boneData.jointDef;
                float angle = Mathf.Max(Mathf.Abs(def.lostop2), Mathf.Abs(def.histop2));
                minAngle[paramIndex] = -angle;
                maxAngle[paramIndex] = angle;
                paramIndex++;
            }
            if (boneData.numFreedoms > 2)
            {
                JointDefinition def = boneData.jointDef;
                float angle = Mathf.Max(Mathf.Abs(def.lostop3), Mathf.Abs(def.histop3));
                minAngle[paramIndex] = -angle;
                maxAngle[paramIndex] = angle;
                paramIndex++;
            }
        }
    }
    
    Dictionary<Transform, BonelessTransformData> getBonelessTransformStates()
    {
        Dictionary<Transform, BonelessTransformData> states = new Dictionary<Transform, BonelessTransformData>();
        getBonelessTransformStatesFrom(targetCharacter, states);
        return states;
    }

    void getBonelessTransformStatesFrom(Transform transform, Dictionary<Transform, BonelessTransformData> states)
    {
        if (!startTransformToBoneData.ContainsKey(transform))
        {
            states.Add(transform, new BonelessTransformData()
            {
                localPosition = transform.localPosition,
                localRotation = transform.localRotation
            });
        }

        for (int i = 0; i < transform.childCount; i++)
        {
            Transform child = transform.GetChild(i);
            getBonelessTransformStatesFrom(child, states);
        }
    }

    void resetBonelessTransforms()
    {
        setBonelessTransforms(bonelessTransformDefaults);
    }

    void setBonelessTransforms(Dictionary<Transform, BonelessTransformData> states)
    {
        foreach (KeyValuePair<Transform, BonelessTransformData> kvp in states)
        {
            Transform transform = kvp.Key;
            BonelessTransformData data = kvp.Value;

            transform.localPosition = data.localPosition;
            transform.localRotation = data.localRotation;
        }
    }
    
}
//} //AaltoGames

