;
; gen_grid_files_cesm_pop.ncl
;
; generate grid files, containing grid_weight and region_mask, from a CESM POP hist (tavg) file
;

begin
    grid = "gx3v7"
    case = "c.e20.C.T62_g37.NK.001"
    hist_dir = "/glade/scratch/klindsay/archive/"+case+"/ocn/hist/"
    hist_fname = systemfunc("ls "+hist_dir+case+".pop.h.[0-9][0-9][0-9][0-9].nc | head -n 1")
    print("creating "+grid+" grid files from "+hist_fname)

    setfileoption("nc", "Format", "NetCDF4Classic")
    setfileoption("nc", "CompressionLevel", 1)

    fptr = addfile(hist_fname, "r")
    KMT = fptr->KMT
    TAREA = fptr->TAREA
    dz = fptr->dz
    z_t = fptr->z_t
    REGION_MASK = fptr->REGION_MASK

    TLAT = fptr->TLAT
    TLONG = fptr->TLONG

    tmp = dimsizes(KMT)
    imt = tmp(1)
    jmt = tmp(0)
    km = dimsizes(dz)

    dz_m = 0.01 * dz
    TAREA_m2 = 0.01 ^ 2 * TAREA

    region_mask = new((/ km, jmt, imt /), integer)
    grid_weight = new((/ km, jmt, imt /), double)
    do k = 0, km-1
        region_mask(k,:,:) = where(k .lt. KMT .and. REGION_MASK .gt. 0, 1, 0)
        region_mask(k,:,:) = where(k .lt. KMT .and. REGION_MASK .eq. -13, 2, region_mask(k,:,:))
        grid_weight(k,:,:) = dz_m(k) * where(region_mask(k,:,:) .gt. 0, TAREA_m2(:,:), 0.0)
    end do

    region_mask!0 = "z_t"
    region_mask&z_t = z_t
    region_mask!1 = "nlat"
    region_mask!2 = "nlon"
    region_mask@coordinates = "TLONG TLAT"
    region_mask@long_name = "Region Mask (1=open ocean, 2=Black Sea, 0=other)"
    region_mask@units = "1"

    DYN_REGMASK = region_mask(0,:,:)

    grid_weight!0 = "z_t"
    grid_weight&z_t = z_t
    grid_weight!1 = "nlat"
    grid_weight!2 = "nlon"
    grid_weight@coordinates = "TLONG TLAT"
    grid_weight@long_name = "Ocean Grid-Cell Volume"
    grid_weight@units = "m3"

    grid_fname = "grid_vars_cesm_pop_"+grid+".nc"
    system("rm -f "+grid_fname)
    fptr = addfile(grid_fname, "c")

    fptr->z_t = z_t
    fptr->TLAT = TLAT
    fptr->TLONG = TLONG
    fptr->region_mask = region_mask
    fptr->DYN_REGMASK = DYN_REGMASK
    fptr->grid_weight = grid_weight
end
