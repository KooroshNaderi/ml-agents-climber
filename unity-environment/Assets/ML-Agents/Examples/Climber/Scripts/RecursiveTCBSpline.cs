using UnityEngine;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;


/*
 * See:
 * http://www.geometrictools.com/Documentation/KBSplines.pdf
 * http://www.gamedev.net/topic/500802-tcb-spline-interpolation/
 * */

public class RecursiveTCBSpline
{
    protected float current = 0, dCurrent = 0;
    public float savedCurrent,savedDCurrent;
    //Default tcb values 0 => catmull-rom spline
    public float t=0;
    const float c=0;   //const, because we need to have tangent continuity (incoming = outgoing) for the recursion to work as implemented
    public float b=0;
    public float linearMix = 0;
    public void setState(float current, float dCurrent)
    {
        this.current=current;
        this.dCurrent=dCurrent;
    }
    public void setCurrentValue(float current)
    {
        this.current = current;
    }
    //incoming tangent at p1, based on three points p0,p1,p2 and their times t0,t1,t2
    float TCBIncomingTangent(float p0, float t0, float p1, float t1, float p2, float t2)
    {
    	return 0.5f*(1-t)*(1+c)*(1-b)*(p2-p1)/(t2-t1) + 0.5f*(1-t)*(1-c)*(1+b)*(p1-p0)/(t1-t0);
    }

    float TCBOutgoingTangent(float p0, float t0, float p1, float t1, float p2, float t2)
    {
    	return 0.5f*(1-t)*(1-c)*(1-b)*(p2-p1)/(t2-t1) + 0.5f*(1-t)*(1+c)*(1+b)*(p1-p0)/(t1-t0);
    }

    // p0 y coordinate of the current point
    // p1 y coordinate of the next point
    // s interpolation value, (t-t0)/timeDelta
    // to0 tangent out of (p0)
    // ti1 tangent in of (p1)
    float InterpolateHermitian(float p0, float p1, float s, float to0, float ti1, float timeDelta)
    {
        float s2=s*s;
        float s3=s2*s;
        //the tcb formula using a Hermite interpolation basis from Eberly
        return (2*s3-3*s2+1)*p0 + (-2*s3+3*s2)*p1 + (s3-2*s2+s)*timeDelta*to0 + (s3-s2)*timeDelta*ti1;
    }

    float InterpolateHermitianTangent(float p0, float p1, float s, float to0, float ti1, float timeDelta)
    {
        float s2=s*s;
        return (6 * s2 - 6 * s) * p0 + (-6 * s2 + 6 * s) * p1 + (3 * s2 - 4 * s + 1) * timeDelta * to0 + (3 * s2 - 2 * s) * timeDelta * ti1;
    }

    //Step the curve forward using the internally stored current value-tangent pair current, dCurrent, and the next two points p1,p2 at times t1,t2. 
    //Returns the remaining time step for the next segment
    public void step(float timeStep, float p1, float t1, float p2, float t2)
    {
        float epsilon = 1e-10f;
        t1 = Mathf.Max(epsilon, t1);    //to prevent NaN
        t2 = Mathf.Max(epsilon, t2);    //to prevent NaN
        float newLinearVal = current + (p1 - current) * timeStep / t1;
        float newLinearTangent = (p1 - current) / t1;
        if (linearMix >= 1)
        {
            current = newLinearVal;
            dCurrent = newLinearTangent;
        }
        else
        {
            float p1IncomingTangent = TCBIncomingTangent(current, 0, p1, t1, p2, t2);
            float newTCBVal = InterpolateHermitian(current, p1, timeStep / t1, dCurrent, p1IncomingTangent, t1);
            float newTCBTangent = InterpolateHermitianTangent(current, p1, timeStep / t1, dCurrent, p1IncomingTangent, t1) / t1;
            current = linearMix * newLinearVal + (1.0f - linearMix) * newTCBVal;
            dCurrent = linearMix * newLinearTangent + (1.0f - linearMix) * newTCBTangent;
        }
    }
    public float currentValue
    {
        get
        {
            return current;
        }
    }
    public float currentDerivativeValue
    {
        get
        {
            return dCurrent;
        }
    }
    public void save()
    {
        savedCurrent = current;
        savedDCurrent = dCurrent;
    }
    public void restore()
    {
        current = savedCurrent;
        dCurrent = savedDCurrent;
    }
    public void copyStateFrom(RecursiveTCBSpline src)
    {
        current = src.current;
        dCurrent = src.dCurrent;
    }
}


public class ScrollingSpline
{
    public int nControlPoints;
    public Vector2[] valuesAndTimes;
    float currentTime, evalStartTime;
    int controlPointIdx;
    public RecursiveTCBSpline spline=new RecursiveTCBSpline();
    public ScrollingSpline(int nControlPoints)
    {
        this.nControlPoints = nControlPoints;
        valuesAndTimes = new Vector2[nControlPoints];
        currentTime = 0;
        evalStartTime = 0;
        controlPointIdx = 0;
    }
    public bool advanceEvalTime(float timeStep)
    {
        float timeToNext=valuesAndTimes[controlPointIdx].y-currentTime;
        float partialStep=Mathf.Min(timeToNext,timeStep);
        spline.step(partialStep, valuesAndTimes[controlPointIdx].x, valuesAndTimes[controlPointIdx].y-currentTime, valuesAndTimes[controlPointIdx + 1].x, valuesAndTimes[controlPointIdx+1].y-currentTime);
        currentTime += partialStep;
        if (timeStep >= timeToNext)
        {
            controlPointIdx++;
            if (controlPointIdx > nControlPoints - 2)
            {
                return false;
            }
            float remaining = timeStep - partialStep;
            if (remaining > 0.0001f)
            {
                spline.step(partialStep, valuesAndTimes[controlPointIdx].x, valuesAndTimes[controlPointIdx].y - currentTime, valuesAndTimes[controlPointIdx + 1].x, valuesAndTimes[controlPointIdx + 1].y - currentTime);
            }
            currentTime += remaining;
        }
        return true;
    }
    public float eval()
    {
        return spline.currentValue;
    }
    public float evalTime()
    {
        return currentTime;
    }
    public void setState(float current, float dCurrent)
    {
        spline.setState(current,dCurrent);
        spline.save();
    }
    public void resetEvalTime()
    {
        spline.restore();
        currentTime = 0;
        controlPointIdx = 0;
    }
    //This will shift all control points closer in time. If the first control point time becomes <= 0, it is removed and the last control point is duplicated (and true is returned instead of false)
    //Returns the number of control points that were duplicated (the caller may want to reinitialize those). Return value of 0 indicates that no points were duplicated
    public int shift(float timeStep)
    {
        //to make the evaluation work, we first have to make the next evaluation start from one timestep ahead
        resetEvalTime();
        advanceEvalTime(timeStep);
        spline.save();

        //now the actual 
        for (int i = 0; i < nControlPoints; i++)
        {
            valuesAndTimes[i].y -= timeStep;
        }
        int result = 0;
        while (valuesAndTimes[0].y <= 0)
        {
            result++;
            for (int i = 0; i < nControlPoints - 1; i++)
            {
                valuesAndTimes[i] = valuesAndTimes[i + 1];
            }
        }
        return result;
    }
}