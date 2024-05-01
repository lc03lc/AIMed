from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
import nibabel as nib
from Processing.Gaussian3D import apply_gaussian_filter
from Processing.Median3D import apply_median_filter
from Processing.DAE3D import apply_auto_denoise
from Segmentation.Seg2D import apply_2D_segmentation
from Segmentation.Seg3D import apply_3D_segmentation
import tempfile
import os

app = FastAPI()


@app.post("/apply_gaussian_filter/")
# 处理高斯滤波请求
async def upload_gaussian_file(file: UploadFile = File(...), sigma: float = 1.0):
    if file.filename.endswith('.nii.gz'):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        filtered_image = apply_gaussian_filter(tmp_path, sigma=sigma)

        output_path = tempfile.mktemp(suffix='.nii.gz')
        nib.save(filtered_image, output_path)

        os.remove(tmp_path)

        return FileResponse(path=output_path, filename=f"filtered_{file.filename}", media_type='application/gzip')
    else:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .nii.gz file.")


@app.post("/apply_median_filter/")
# 处理中值滤波请求
async def upload_and_median_filter(file: UploadFile = File(...), size: int = 3):
    if file.filename.endswith('.nii.gz'):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        filtered_image = apply_median_filter(tmp_path, size=size)

        output_path = tempfile.mktemp(suffix='.nii.gz')
        nib.save(filtered_image, output_path)

        os.remove(tmp_path)

        return FileResponse(path=output_path, filename=f"median_filtered_{file.filename}",
                            media_type='application/gzip')
    else:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .nii.gz file.")


@app.post("/apply_auto_denoise/")
# 处理自动编码降噪请求
async def upload_auto_denoise(file: UploadFile = File(...)):
    if file.filename.endswith('.nii.gz'):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        denoised_image = apply_auto_denoise(tmp_path)

        output_path = tempfile.mktemp(suffix='.nii.gz')
        nib.save(denoised_image, output_path)

        os.remove(tmp_path)

        return FileResponse(path=output_path, filename=f"denoised_{file.filename}", media_type='application/gzip')
    else:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .nii.gz file.")


@app.post("/apply_2D_segmentation/")
# 处理二维分割请求
async def upload_2D_segmentation(
        file: UploadFile = File(...),
        xmin: int = Form(...),
        ymin: int = Form(...),
        xmax: int = Form(...),
        ymax: int = Form(...),
        step: int = Form(...)
):
    if not file.filename.endswith('.jpg'):
        raise HTTPException(status_code=400, detail="Invalid file type. Only .jpg files are accepted.")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    mask_image, cutout_image, outline_image = apply_2D_segmentation(tmp_path, xmin, ymin, xmax, ymax, step)

    os.remove(tmp_path)

    return {
        "mask_image": FileResponse(mask_image),
        "cutout_image": FileResponse(cutout_image),
        "outline_image": FileResponse(outline_image)
    }


@app.post("/apply_3D_segmentation/")
# 处理三维分割请求
async def upload_3D_segmentation(input_file: UploadFile = File(...), label: str = Form(...), modality: str = Form(...)):
    if not input_file.filename.endswith('.nii.gz'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .nii.gz file.")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp:
        contents = await input_file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    output_file_path = apply_3D_segmentation(tmp_path, label, modality)

    os.remove(tmp_path)

    return FileResponse(path=output_file_path, filename=f"segmented_{input_file.filename}",
                        media_type='application/gzip')




