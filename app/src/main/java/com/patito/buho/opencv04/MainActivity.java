package com.patito.buho.opencv04;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.SurfaceView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;


public class MainActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "MainActivity";
    private CameraBridgeViewBase mOpenCvCameraView;

    Mat mRgba, mGray, mIntermediateMat, mContours;

    private MatOfPoint2f mMOP2f1, mMOP2f2;

    private int iContourAreaMin = 1000, iLineThickness = 3;

    private Scalar colorRed, colorGreen;
    private Size sSize, sSize3, sSize5, sMatSize;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {

            switch (status){
                case LoaderCallbackInterface.SUCCESS:
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    @Override
    protected void onResume() {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_9, this, mLoaderCallback);
    }

    @Override
    protected void onPause() {
        super.onPause();
        if ( mOpenCvCameraView != null )
            mOpenCvCameraView.disableView();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if ( mOpenCvCameraView != null )
            mOpenCvCameraView.disableView();
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        mOpenCvCameraView  = (CameraBridgeViewBase) findViewById(R.id.HelloOpenCvView);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }


    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat();
        mGray = new Mat();
        mIntermediateMat = new Mat();
        mContours = new Mat();

        mMOP2f1 = new MatOfPoint2f();
        mMOP2f2 = new MatOfPoint2f();

        colorRed = new Scalar(255, 0, 0, 255);
        colorGreen = new Scalar(0, 255, 0, 255);

        sMatSize = new Size();
        sSize = new Size();
        sSize3 = new Size(3, 3);
        sSize5 = new Size(5, 5);
    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
        mGray.release();
        mIntermediateMat.release();
        mContours.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        Mat src = inputFrame.rgba();
        Mat bw = new Mat();

        Imgproc.cvtColor(src, bw, Imgproc.COLOR_BGR2BGRA);
        Imgproc.GaussianBlur(bw, bw, new Size(1,1), 0, 0, Imgproc.BORDER_TRANSPARENT);
        Imgproc.Canny(bw, bw, 35, 75, 3, false);

        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();

        contours.clear();
        Imgproc.findContours(bw, contours, mContours, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);


        for (int x = 0; x < contours.size(); x++) {

            double d = Imgproc.contourArea(contours.get(x));

            long total = contours.get(x).total();

            // get an approximation of the contour (last but one param is the min required
            // distance between the real points and the new approximation (in pixels)

            // contours is a List<matofpoint>
            // so contours.get(x) is a single MatOfPoint
            // but to use approxPolyDP we need to pass a MatOfPoint2f
            // so we need to do a conversion
            contours.get(x).convertTo(mMOP2f1, CvType.CV_32FC2);

//            iContourAreaMin = 10;
            if (d > iContourAreaMin) {

                double epsilon =  total * 0.4;  //130;

                Imgproc.approxPolyDP(mMOP2f1, mMOP2f2, epsilon , true);

                // convert back to MatOfPoint and put it back in the list
                mMOP2f2.convertTo(contours.get(x), CvType.CV_32S);

                if ( mMOP2f2.total() == 4 ){

                    // draw the contour itself
                    Imgproc.drawContours(src, contours, x, colorRed, iLineThickness);

                    List<Point> points = mMOP2f2.toList();
                    for (Point point : points) {
                        double x3 = point.x;
                        double y3 = point.y;
                        Log.d(TAG, "X encontrado :: " + x3);
                        Log.d(TAG, "Y encontrado :: " + y3);
                        Core.circle(src, point, 15, colorGreen, iLineThickness - 1);
                        DrawCross (src, colorGreen, point);

                    }

                    // data for analyzer
                    Log.d(TAG, "====== total de contornos :: " + String.valueOf(total));
                    Log.d(TAG, "====== area detectada :: " + String.valueOf(d));
                    Log.d(TAG, "====== area parcial :: " + String.valueOf(epsilon));

                    // get perspective, optional :: it is necessary adjust to support real sizes, but work
                    MatOfPoint2f dst = new MatOfPoint2f();
                    dst.push_back(new MatOfPoint(new Point(0,0)));
                    dst.push_back(new MatOfPoint(new Point(src.cols() - 1,0)));
                    dst.push_back(new MatOfPoint(new Point(src.cols() - 1,src.rows() - 1)));
                    dst.push_back(new MatOfPoint(new Point(0,src.rows() - 1)));

                    dst.convertTo(dst, CvType.CV_32F);
                    mMOP2f2.convertTo(mMOP2f2, CvType.CV_32F);

                    Mat warp_dst = new Mat( src.rows(), src.cols(), src.type() );
                    Mat perspectiveTransform = Imgproc.getPerspectiveTransform(mMOP2f2, dst);
                    Imgproc.warpPerspective(src, src, perspectiveTransform, warp_dst.size(), Imgproc.INTER_CUBIC);
                }
            }
        }

        return src;
    }

    public void DrawCross (Mat mat, Scalar color, Point pt) {
        int iCentreCrossWidth = 24;

        Point pt1 = new Point();
        Point pt2 = new Point();
        pt1.x = pt.x - (iCentreCrossWidth >> 1);
        pt1.y = pt.y;
        pt2.x = pt.x + (iCentreCrossWidth >> 1);
        pt2.y = pt.y;

        Core.line(mat, pt1, pt2, color, iLineThickness - 1);

        pt1.x = pt.x;
        pt1.y = pt.y + (iCentreCrossWidth >> 1);
        pt2.x = pt.x;
        pt2.y = pt.y  - (iCentreCrossWidth >> 1);

        Core.line(mat, pt1, pt2, color, iLineThickness - 1);

    }
}
