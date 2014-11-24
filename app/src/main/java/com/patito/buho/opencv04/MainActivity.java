package com.patito.buho.opencv04;

import android.app.Activity;
import android.content.ContentValues;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
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
import org.opencv.core.Range;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.math.BigDecimal;
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
//                    Imgproc.drawContours(src, contours, x, colorRed, iLineThickness);

                    List<Point> points = mMOP2f2.toList();
                    for (Point point : points) {
                        double x3 = point.x;
                        double y3 = point.y;
                        Log.d(TAG, "X encontrado :: " + x3);
                        Log.d(TAG, "Y encontrado :: " + y3);
//                        Core.circle(src, point, 15, colorGreen, iLineThickness - 1);
//                        DrawCross (src, colorGreen, point);

                    }

                    // data for analyzer
                    Log.d(TAG, "====== total de contornos :: " + String.valueOf(total));
                    Log.d(TAG, "====== area detectada :: " + String.valueOf(d));
                    Log.d(TAG, "====== area parcial :: " + String.valueOf(epsilon));

                    // Get mass center
                    Point center = new Point(0,0);
                    for (Point point : points) {
                        center.x += point.x;
                        center.y += point.y;
                    }
                    center.x *= (1.0 / points.size());
                    center.y *= (1.0 / points.size());

                    // Get ordered points
                    List<Point> pointList = sortCorners(points, center);

                    for (Point point : pointList) { // p1(82, 111) p2(811, 113) p3(841, 410) p4(77, 440)
                        Log.d(TAG, "X ordenado encontrado :: " + point.x);
                        Log.d(TAG, "Y ordenada encontrado :: " + point.y);
                    }
                    Log.d(TAG, "src.rows() :: " + src.rows());
                    Log.d(TAG, "src.cols() :: " + src.cols());


                    // get the max distance between same points
                    double dist_x1 = pointList.get(1).x - pointList.get(0).x;
                    double dist_x2 = pointList.get(2).x - pointList.get(3).x;
                    double disX = dist_x1 > dist_x2 ? dist_x1:dist_x2;

                    double dist_y1 = pointList.get(3).y - pointList.get(0).y;
                    double dist_y2 = pointList.get(2).y - pointList.get(1).y;
                    double disY = dist_y1 > dist_y2 ? dist_y1:dist_y2;

                    Log.d(TAG, " aspect relation -> " + disX / disY);

                    // get perspective, optional :: it is necessary adjust to support real sizes, but work
                    MatOfPoint2f dst = new MatOfPoint2f();
                    dst.push_back(new MatOfPoint(new Point(0,0)));
                    dst.push_back(new MatOfPoint(new Point(disX, 0)));
                    dst.push_back(new MatOfPoint(new Point(disX, disY)));
                    dst.push_back(new MatOfPoint(new Point(0, disY)));

                    MatOfPoint2f originalPoints = new MatOfPoint2f();

                    // src.cols() -> x
                    // src.rows() -> y
                    originalPoints.push_back(new MatOfPoint(pointList.get(0)));
                    originalPoints.push_back(new MatOfPoint(pointList.get(1)));
                    originalPoints.push_back(new MatOfPoint(pointList.get(2)));
                    originalPoints.push_back(new MatOfPoint(pointList.get(3)));

                    dst.convertTo(dst, CvType.CV_32F);
                    originalPoints.convertTo(originalPoints, CvType.CV_32F);

                    Mat warp_dst = new Mat( src.rows(), src.cols(), src.type() );
                    Mat dest = new Mat( dst.rows(), dst.cols(), dst.type() );

                    Mat perspectiveTransform = Imgproc.getPerspectiveTransform(originalPoints, dst);
                    Imgproc.warpPerspective(src, dest, perspectiveTransform, warp_dst.size(), Imgproc.INTER_NEAREST);


                    int length = new BigDecimal(disX).intValue();
                    int higth = new BigDecimal(disY).intValue();

                    Rect roi = new Rect(0, 0, length, higth);
                    Mat cheque = dest.submat(roi);

                    takePhoto(cheque);

                    return dest;
                }
            }
        }

        return src;
    }

    private List<Point> sortCorners(List<Point> corners, Point center) {

        List<Point> top = new ArrayList<Point>(), bot = new ArrayList<Point>();
        for (Point corner : corners) {
            if (corner.y < center.y)
                top.add(corner);
            else
                bot.add(corner);
        }
        List<Point> corners_sorted = new ArrayList<Point>();
        if (top.size() == 2 && bot.size() == 2){
            Point tl = top.get(0).x > top.get(1).x ? top.get(1) : top.get(0);
            Point tr = top.get(0).x > top.get(1).x ? top.get(0) : top.get(1);
            Point bl = bot.get(0).x > bot.get(1).x ? bot.get(1) : bot.get(0);
            Point br = bot.get(0).x > bot.get(1).x ? bot.get(0) : bot.get(1);
            corners_sorted.add(tl);
            corners_sorted.add(tr);
            corners_sorted.add(br);
            corners_sorted.add(bl);
        }

        return corners_sorted;
    }

    private void DrawCross (Mat mat, Scalar color, Point pt) {
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

    private void takePhoto(final Mat rgba) {
        // determine  the path and metadata for the photo
        final long currentTimeMillis = System.currentTimeMillis();
        final String app_name = getString(R.string.app_name);
        final String galleryPath = Environment.getExternalStoragePublicDirectory(Environment
                .DIRECTORY_PICTURES)
                .toString();
        final String albumPath = galleryPath + "/" + app_name;
        final String photoPath = albumPath + "/" + currentTimeMillis + ".png";
        final ContentValues values = new ContentValues();
        values.put(MediaStore.MediaColumns.DATA, photoPath);
        values.put(MediaStore.Images.Media.MIME_TYPE, "image/png");
        values.put(MediaStore.Images.Media.TITLE, app_name);
        values.put(MediaStore.Images.Media.DESCRIPTION, app_name);
        values.put(MediaStore.Images.Media.DATE_TAKEN, currentTimeMillis);

        // Ensure that the album directory exist
        File album = new File(albumPath);
        if ( !album.isDirectory() && !album.mkdirs() ){
            Log.e(TAG, "Failed to create album directory at " + albumPath);
//            onTakePhotoFailed();
            return;
        }

        // try to create the photo
        Mat mBgr = new Mat();
        Imgproc.cvtColor(rgba, mBgr, Imgproc.COLOR_BGR2BGRA, 3);
        if ( !Highgui.imwrite(photoPath, mBgr)) {
            Log.e(TAG, "Failed to save photo to " + photoPath);
//            onTakePhotoFailed();
            return;
        }

        Log.d(TAG, "Photo saved successfully to " + photoPath);

        // try to insert the photo into the MediaStore
        Uri uri;
        try {
            uri = getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
        } catch (final Exception e) {
            Log.e(TAG, "Failed to insert photo into MediaStore", e);

            // Since the insertion failed, delete the photo
            File photo = new File(photoPath);
            if ( !photo.delete() )
                Log.e(TAG, "Failed to delete non-inserted photo");

//            onTakePhotoFailed();
            return;
        }

        // open the photo in LabActivity
        /*final Intent intent = new Intent(this, LabActivity.class);
        intent.putExtra(LabActivity.EXTRA_PHOTO_URI, uri);
        intent.putExtra(LabActivity.EXTRA_PHOTO_DATA_PATH, photoPath);
        startActivity(intent);*/
    }
}
