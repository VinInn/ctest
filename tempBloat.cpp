/* --------------------------------------------------
   Test program to give a rough measure of "template bloat."
   If the macro "DIFFERENT" is defined at compile-time, this
   program creates a list<T*> for 100 different types of T.
   Otherwise, it creates 100 instances of a list of a single
   pointer type.
   A capable compiler will recognise that the binary representation
   of list<T*> is the same for all T and it need retain only a
   single copy of the instantiation code in the program.
   --------------------------------------------------*/
#include <list>
class  x0;
class  x1;
class  x2;
class  x3;
class  x4;
class  x5;
class  x6;
class  x7;
class  x8;
class  x9;
class x10;
class x11;
class x12;
class x13;
class x14;
class x15;
class x16;
class x17;
class x18;
class x19;
class x20;
class x21;
class x22;
class x23;
class x24;
class x25;
class x26;
class x27;
class x28;
class x29;
class x30;
class x31;
class x32;
class x33;
class x34;
class x35;
class x36;
class x37;
class x38;
class x39;
class x40;
class x41;
class x42;
class x43;
class x44;
class x45;
class x46;
class x47;
class x48;
class x49;
class x50;
class x51;
class x52;
class x53;
class x54;
class x55;
class x56;
class x57;
class x58;
class x59;
class x60;
class x61;
class x62;
class x63;
class x64;
class x65;
class x66;
class x67;
class x68;
class x69;
class x70;
class x71;
class x72;
class x73;
class x74;
class x75;
class x76;
class x77;
class x78;
class x79;
class x80;
class x81;
class x82;
class x83;
class x84;
class x85;
class x86;
class x87;
class x88;
class x89;
class x90;
class x91;
class x92;
class x93;
class x94;
class x95;
class x96;
class x97;
class x98;
class x99;
int main()
{
#if defined DIFFERENT // create 100 lists of different pointer types
  std::list<x0*> v0;
  std::list<x1*> v1;
  std::list<x2*> v2;
  std::list<x3*> v3;
  std::list<x4*> v4;
  std::list<x5*> v5;
  std::list<x6*> v6;
  std::list<x7*> v7;
  std::list<x8*> v8;
  std::list<x9*> v9;
  std::list<x10*> v10;
  std::list<x11*> v11;
  std::list<x12*> v12;
  std::list<x13*> v13;
  std::list<x14*> v14;
  std::list<x15*> v15;
  std::list<x16*> v16;
  std::list<x17*> v17;
  std::list<x18*> v18;
  std::list<x19*> v19;
  std::list<x20*> v20;
  std::list<x21*> v21;
  std::list<x22*> v22;
  std::list<x23*> v23;
  std::list<x24*> v24;
  std::list<x25*> v25;
  std::list<x26*> v26;
  std::list<x27*> v27;
  std::list<x28*> v28;
  std::list<x29*> v29;
  std::list<x30*> v30;
  std::list<x31*> v31;
  std::list<x32*> v32;
  std::list<x33*> v33;
  std::list<x34*> v34;
  std::list<x35*> v35;
  std::list<x36*> v36;
  std::list<x37*> v37;
  std::list<x38*> v38;
  std::list<x39*> v39;
  std::list<x40*> v40;
  std::list<x41*> v41;
  std::list<x42*> v42;
  std::list<x43*> v43;
  std::list<x44*> v44;
  std::list<x45*> v45;
  std::list<x46*> v46;
  std::list<x47*> v47;
  std::list<x48*> v48;
  std::list<x49*> v49;
  std::list<x50*> v50;
  std::list<x51*> v51;
  std::list<x52*> v52;
  std::list<x53*> v53;
  std::list<x54*> v54;
  std::list<x55*> v55;
  std::list<x56*> v56;
  std::list<x57*> v57;
  std::list<x58*> v58;
  std::list<x59*> v59;
  std::list<x60*> v60;
  std::list<x61*> v61;
  std::list<x62*> v62;
  std::list<x63*> v63;
  std::list<x64*> v64;
  std::list<x65*> v65;
  std::list<x66*> v66;
  std::list<x67*> v67;
  std::list<x68*> v68;
  std::list<x69*> v69;
  std::list<x70*> v70;
  std::list<x71*> v71;
  std::list<x72*> v72;
  std::list<x73*> v73;
  std::list<x74*> v74;
  std::list<x75*> v75;
  std::list<x76*> v76;
  std::list<x77*> v77;
  std::list<x78*> v78;
  std::list<x79*> v79;
  std::list<x80*> v80;
  std::list<x81*> v81;
  std::list<x82*> v82;
  std::list<x83*> v83;
  std::list<x84*> v84;
  std::list<x85*> v85;
  std::list<x86*> v86;
  std::list<x87*> v87;
  std::list<x88*> v88;
  std::list<x89*> v89;
  std::list<x90*> v90;
  std::list<x91*> v91;
  std::list<x92*> v92;
  std::list<x93*> v93;
  std::list<x94*> v94;
  std::list<x95*> v95;
  std::list<x96*> v96;
  std::list<x97*> v97;
  std::list<x98*> v98;
  std::list<x99*> v99;
#else            // create 100 instances of a single list<T*> type
  std::list<x0*> v0;
  std::list<x0*> v1;
  std::list<x0*> v2;
  std::list<x0*> v3;
  std::list<x0*> v4;
  std::list<x0*> v5;
  std::list<x0*> v6;
  std::list<x0*>  v7;
  std::list<x0*>  v8;
  std::list<x0*>  v9;
  std::list<x0*> v10;
  std::list<x0*> v11;
  std::list<x0*> v12;
  std::list<x0*> v13;
  std::list<x0*> v14;
  std::list<x0*> v15;
  std::list<x0*> v16;
  std::list<x0*> v17;
  std::list<x0*> v18;
  std::list<x0*> v19;
  std::list<x0*> v20;
  std::list<x0*> v21;
  std::list<x0*> v22;
  std::list<x0*> v23;
  std::list<x0*> v24;
  std::list<x0*> v25;
  std::list<x0*> v26;
  std::list<x0*> v27;
  std::list<x0*> v28;
  std::list<x0*> v29;
  std::list<x0*> v30;
  std::list<x0*> v31;
  std::list<x0*> v32;
  std::list<x0*> v33;
  std::list<x0*> v34;
  std::list<x0*> v35;
  std::list<x0*> v36;
  std::list<x0*> v37;
  std::list<x0*> v38;
  std::list<x0*> v39;
  std::list<x0*> v40;
  std::list<x0*> v41;
  std::list<x0*> v42;
  std::list<x0*> v43;
  std::list<x0*> v44;
  std::list<x0*> v45;
  std::list<x0*> v46;
  std::list<x0*> v47;
  std::list<x0*> v48;
  std::list<x0*> v49;
  std::list<x0*> v50;
  std::list<x0*> v51;
  std::list<x0*> v52;
  std::list<x0*> v53;
  std::list<x0*> v54;
  std::list<x0*> v55;
  std::list<x0*> v56;
  std::list<x0*> v57;
  std::list<x0*> v58;
  std::list<x0*> v59;
  std::list<x0*> v60;
  std::list<x0*> v61;
  std::list<x0*> v62;
  std::list<x0*> v63;
  std::list<x0*> v64;
  std::list<x0*> v65;
  std::list<x0*> v66;
  std::list<x0*> v67;
  std::list<x0*> v68;
  std::list<x0*> v69;
  std::list<x0*> v70;
  std::list<x0*> v71;
  std::list<x0*> v72;
  std::list<x0*> v73;
  std::list<x0*> v74;
  std::list<x0*> v75;
  std::list<x0*> v76;
  std::list<x0*> v77;
  std::list<x0*> v78;
  std::list<x0*> v79;
  std::list<x0*> v80;
  std::list<x0*> v81;
  std::list<x0*> v82;
  std::list<x0*> v83;
  std::list<x0*> v84;
  std::list<x0*> v85;
  std::list<x0*> v86;
  std::list<x0*> v87;
  std::list<x0*> v88;
  std::list<x0*> v89;
  std::list<x0*> v90;
  std::list<x0*> v91;
  std::list<x0*> v92;
  std::list<x0*> v93;
  std::list<x0*> v94;
  std::list<x0*> v95;
  std::list<x0*> v96;
  std::list<x0*> v97;
  std::list<x0*> v98;
  std::list<x0*> v99;
#endif
  int s = 
  v0.size()+
  v1.size()+
  v2.size()+
  v3.size()+
  v4.size()+
  v5.size()+
  v6.size()+
  v7.size()+
  v8.size()+
  v9.size()+
  v10.size()+
  v11.size()+
  v12.size()+
  v13.size()+
  v14.size()+
  v15.size()+
  v16.size()+
  v17.size()+
  v18.size()+
  v19.size()+
  v20.size()+
  v21.size()+
  v22.size()+
  v23.size()+
  v24.size()+
  v25.size()+
  v26.size()+
  v27.size()+
  v28.size()+
  v29.size()+
  v30.size()+
  v31.size()+
  v32.size()+
  v33.size()+
  v34.size()+
  v35.size()+
  v36.size()+
  v37.size()+
  v38.size()+
  v39.size()+
  v40.size()+
  v41.size()+
  v42.size()+
  v43.size()+
  v44.size()+
  v45.size()+
  v46.size()+
  v47.size()+
  v48.size()+
  v49.size()+
  v50.size()+
  v51.size()+
  v52.size()+
  v53.size()+
  v54.size()+
  v55.size()+
  v56.size()+
  v57.size()+
  v58.size()+
  v59.size()+
  v60.size()+
  v61.size()+
  v62.size()+
  v63.size()+
  v64.size()+
  v65.size()+
  v66.size()+
  v67.size()+
  v68.size()+
  v69.size()+
  v70.size()+
  v71.size()+
  v72.size()+
  v73.size()+
  v74.size()+
  v75.size()+
  v76.size()+
  v77.size()+
  v78.size()+
  v79.size()+
  v80.size()+
  v81.size()+
  v82.size()+
  v83.size()+
  v84.size()+
  v85.size()+
  v86.size()+
  v87.size()+
  v88.size()+
  v89.size()+
  v90.size()+
  v91.size()+
  v92.size()+
  v93.size()+
  v94.size()+
  v95.size()+
  v96.size()+
  v97.size()+
  v98.size()+
    v99.size();
  return s;
}
