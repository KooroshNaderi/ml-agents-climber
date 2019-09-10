using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class HumanoidRigControlTest : MonoBehaviour {
    public bool fixCharacterInAir = true;
    public bool savePoseToFile = false;
    public Vector3[] targetCurrentPose;
    HumanoidMecanimRig rig;

    // Use this for initialization
    void Start ()
    {
        Time.fixedDeltaTime = 1 / 100.0f;
        rig = GetComponent<HumanoidMecanimRig>();
        if (rig == null)
            Debug.LogException(new System.Exception("This script needs a HumanoidMecanimRig component on the same game object"));
        rig.initialize();

        targetCurrentPose = new Vector3[rig.numControlDOFs()];
        readPoseAngleFromFile();
    }

    void Update()
    {
        rig.freezeBody(fixCharacterInAir);

        if (savePoseToFile)
        {
            writePoseAngleToFile();
            savePoseToFile = false;
        }
    }

    void FixedUpdate()
    {
        if (targetCurrentPose.Length < rig.numControlDOFs())
            targetCurrentPose = new Vector3[rig.numControlDOFs()];

        float[] targetPose = new float[rig.numControlDOFs()];
        for (int i = 0; i < rig.numControlDOFs(); i++)
        {
            targetPose[i] = targetCurrentPose[i][0];
        }
        rig.driveToPose(targetPose);//, 1f);
        for (int i = 0; i < rig.numControlDOFs(); i++)
        {
            targetCurrentPose[i][0] = targetPose[i];
        }

        float[] readPose = new float[rig.numControlDOFs()];
        rig.poseToMotorAngles(ref readPose);
        for (int i = 0; i < rig.numControlDOFs(); i++)
        {
            targetCurrentPose[i][1] = readPose[i];
        }

        float[] readTorques = new float[rig.numControlDOFs()];
        rig.getMotorAngleRates(ref readTorques);
        for (int i = 0; i < rig.numControlDOFs(); i++)
        {
            targetCurrentPose[i][2] = readTorques[i];
        }

        //      for (int i = 0; i < targetPose.Length; i++)
        //      {
        //          targetPose[i] = Mathf.Clamp(targetPose[i], rig.angleMinLimits()[i], rig.angleMaxLimits()[i]);
        //      }
        //      rig.driveToPose(targetPose, Time.fixedDeltaTime);

        //      Debug.Log("Control effort: " + rig.getControlEffort());
    }

    void writePoseAngleToFile()
    {
        FileStream posefile = new FileStream("climberAnglePos.txt", FileMode.Create);
        StreamWriter writer = new StreamWriter(posefile);
        for (int i = 0; i < rig.numControlDOFs(); i++)
        {
            writer.WriteLine(targetCurrentPose[i][0].ToString());
        }
        writer.Flush();
        writer.Close();
        posefile.Close();
    }

    public void readPoseAngleFromFile()
    { 
        FileStream posefile = new FileStream("climberAnglePos.txt", FileMode.Open);
        StreamReader reader = new StreamReader(posefile);
        for (int i = 0; i < rig.numControlDOFs(); i++)
        {
            string _str = reader.ReadLine();
            targetCurrentPose[i][0] = float.Parse(_str);
        }
        reader.Close();
        posefile.Close();
        return;
    }
}
