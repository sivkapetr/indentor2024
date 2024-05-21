// Created by Liogky Alexey on 09.02.2024.
//

/**
 * This program generates  and solves a finite element system for the stationary linear elasticity problem
 *
 * \f[
 * \begin{aligned}
 *   \mathrm{div}\ \mathbb{P}  + f &= 0\    in\  \Omega  \\
 *          \mathbf{u}             &= \mathbf{u}_0\  on\  \Gamma_D\\
 *     \mathbb{P} \cdot \mathbf{N} &= -p\ \mathrm{adj}\ \mathbb{F}^T \mathbf{N}\ on\  \Gamma_P\\  
 *     \mathbb{P} \cdot \mathbf{N} &= 0\    on\  \Gamma_0\\
 *                      \mathbb{P} &= \mathbb{F} \cdot \mathbb{S}\\ 
 *   \mathbb{F}_{ij} &= \mathbb{I}_{ij} + \mathrm{grad}_j \mathbf{u}_i\\
 *        \mathbb{S} &= \lambda\ \mathrm{tr}\ \mathbb{E}\ \mathbb{I} + 2 \mu \mathbb{E} \\
 *        \mathbb{E} &= \frac{\mathbb{F}^T \cdot \mathbb{F}  - \mathbb{I} } {2} \\
 * \end{aligned}
 * \f]
 * 
 *
 * where where Ω = [0,10] x [0,1]^2, Γ_D = {0}x[0,1]^2, Γ_P = [0,10]x[0,1]x{0}, Γ_0 = ∂Ω \ (Γ_D ⋃ Γ_P)            
 * The user-defined coefficients are 
 *  μ(x)   = 3.0         - first Lame coefficient
 *  λ(x)   = 1.0         - second Lame coefficient
 *  f(x)   = { 0, 0, 0 } - external body forces
 *  u_0(x) = { 0, 0, 0 } - essential (Dirichlet) boundary condition
 *  p(x)   = 0.001         - pressure on bottom part
 * 
 */

#include "prob_args.h"
#include "anifem++/autodiff/cauchy_strain_autodiff.h"

using namespace INMOST;

struct InputArgs1VarNonLin: public InputArgs1Var{
    using ParentType = InputArgs1Var;
    double nlin_rel_err = 1e-8, nlin_abs_err = 1e-8;
    int nlin_maxit = 15;
    double lin_abs_scale = 0.01;

    uint parseArg(int argc, char* argv[], bool print_messages = true) override {
        #define GETARG(X)   if (i+1 < static_cast<uint>(argc)) { X }\
            else { if (print_messages) std::cerr << "ERROR: Not found argument" << std::endl; exit(-1); }
        uint i = 0;
        if (strcmp(argv[i], "-re") == 0 || strcmp(argv[i], "--rel_error") == 0){
            GETARG(nlin_rel_err = std::stod(argv[++i]);)
            return i+1; 
        } if (strcmp(argv[i], "-ae") == 0 || strcmp(argv[i], "--abs_error") == 0){
            GETARG(nlin_abs_err = std::stod(argv[++i]);)
            return i+1; 
        } if (strcmp(argv[i], "-ni") == 0 || strcmp(argv[i], "--maxits") == 0){
            GETARG(nlin_maxit = std::stoi(argv[++i]);)
            return i+1; 
        } if (                               strcmp(argv[i], "--latol_scl") == 0){
            GETARG(lin_abs_scale = std::stod(argv[++i]);)
            return i+1; 
        } else 
            return ParentType::parseArg(argc, argv, print_messages);
        #undef GETARG
    }
    void print(std::ostream& out = std::cout, const std::string& prefix = "") const override{
        out << prefix << "newton: stop tolerances: rel_tol = " << nlin_rel_err << ", abs_tol = " << nlin_abs_err << "\n";
        out << prefix << "newton: maximum iteration number = " << nlin_maxit << "\n";
        out << prefix << "newton: linear solver absolute tolerance scale = " << lin_abs_scale << "\n";
        ParentType::print(out, prefix);
    }

protected:
    void printArgsDescr(std::ostream& out = std::cout, const std::string& prefix = "") override{
        out << prefix << "  -re, --rel_error DVAL    <Set stop relative residual norm for newton method, default=\"" << nlin_rel_err << "\">\n";
        out << prefix << "  -ae, --abs_error DVAL    <Set stop absolute residual norm for newton method, default=\"" << nlin_abs_err << "\">\n";
        out << prefix << "  -ni, --maxits    IVAL    <Set maximum number of newton method iterations, default=\"" << nlin_maxit << "\">\n";
        out << prefix << "       --latol_scl IVAL    <Set newton linear solver absolute tolerance scale, default=\"" << lin_abs_scale << "\">\n";
        ParentType::printArgsDescr(out, prefix);
    }
};

int main(int argc, char* argv[]){
    InputArgs1VarNonLin p;
    std::string prob_name = "nonlin_elast";
    p.save_prefix = prob_name + "_out"; 
    p.parseArgs_mpi(&argc, &argv);
    unsigned quad_order = p.max_quad_order;

    int pRank = 0, pCount = 1;
    InmostInit(&argc, &argv, p.lin_sol_db, pRank, pCount);

    Mesh* m = new Mesh();
    if (pRank == 0) {
        if (argc <= 1)
            m->Load("mesh.msh");
        else
            m->Load(argv[1]);    
    }
    if (pCount > 1){
        Partitioner * p = new Partitioner(m);
        p->SetMethod(Partitioner::INNER_KMEANS, Partitioner::Partition); // выбор разделителя
        p->Evaluate();
        delete p;
        BARRIER
        
        // prior exchange ghost to get optimization in Redistribute
        m->ExchangeGhost(1, NODE); //<- required for parallel fem
        m->Redistribute();
        BARRIER;
        m->ExchangeGhost(1, NODE); //<- required for parallel fem
        m->ReorderEmpty(CELL | FACE | EDGE | NODE);
    }
    m->AssignGlobalID(NODE);
    print_mesh_sizes(m);
    m->ExchangeData(m->GetTag("GMSH_TAGS"), FACE);

    using namespace Ani;
    //generate FEM space from it's name
    FemSpace UFem = choose_space_from_name(p.USpace)^3;
    uint unf = UFem.dofMap().NumDofOnTet();
    auto& dofmap = UFem.dofMap();
    auto mask = GeomMaskToInmostElementType(dofmap.GetGeomMask()) | FACE;
    
    // Set boundary labels on all boundaries
    const int GMSH_INTERNAL_PART = 0, GMSH_FREE_BND = 5, GMSH_PRESSURE_BND = 4, GMSH_FIX_X = 1, GMSH_FIX_Y = 2, GMSH_FIX_Z = 3;
    const int   INTERNAL_PART = 0,                      //0
                FREE_BND = 1<<(GMSH_FREE_BND-1),        //16
                PRESSURE_BND = 1<<(GMSH_PRESSURE_BND-1),//8 
                FIX_X = 1<<(GMSH_FIX_X-1),              //1
                FIX_Y = 1<<(GMSH_FIX_Y-1),              //2
                FIX_Z = 1<<(GMSH_FIX_Z-1);              //4
    
    // Set boundary labels on all boundaries

    Tag gmsh = m->GetTag("GMSH_TAGS");
	Tag BndLabel = m->CreateTag("Label", DATA_INTEGER, mask, NONE, 1);
    for (auto e = m->BeginElement(mask); e != m->EndElement(); ++e) e->Integer(BndLabel) = INTERNAL_PART;
    for (auto f = m->BeginFace(); f != m->EndFace(); ++f){
        if (!f->HaveData(gmsh) || f->IntegerArray(gmsh).empty()) continue;
        int lbl = f->IntegerArray(gmsh)[0];
        // std::cout << "lbl  = " << lbl << std::endl;
        if (lbl == GMSH_INTERNAL_PART) continue;
        lbl = ( 1 << (lbl - 1) ) ;
        auto set_label = [BndLabel, lbl](const auto& elems){
            for (unsigned ni = 0; ni < elems.size(); ni++)
                elems[ni].Integer(BndLabel) |= lbl;
        };
        if (mask & NODE) set_label(f->getNodes());
        if (mask & EDGE) set_label(f->getEdges());
        if (mask & FACE) set_label(f->getFaces());
    }

    //============================================================================================
    // Параметры тут
    //============================================================================================


    
    //Problem parameters
    double H = 10.0e-3, dH = 0.1e-3, Hend = 8.0e-3; // Высота центра индентора - старт, шаг, конец
    double alpha = 1e6;                             // Множитель отталкивающей функции
    double alpha_mul = 1.05;                        // Итерационный множитель множителя
    double r = 5.0e-3;                                // Радиус индентора
    double q = 0.01;                                // Параметр отталкивающей функции

    auto Potential = [](SymMtx3D<> E, unsigned char dif = 3){ // Функция потенциала
        auto I1 = Mech::I1<>{dif, E};               // Первый инвариант
        auto I2 = Mech::I2<>{dif, E};               // Второй инвариант
        auto I3 = Mech::I3<>{dif, E};               // Третий инвариант
        auto J = Mech::J<>{dif, E};                 // Якобиан

        double C1 = 4.702e3;                        // Параметры потенциала
        double C2 = 0.47e3;
        double C3 = 0.11e3;                         
        
        // Разные варианты потенциалов. Последний член отвечает за несжимаемость тела
        //auto W = C1*(I1 - 3) + C2*(I2 - 3) + C1*1000*(I3-2*log(sqrt(I3))-1); // Mooney-Rivlin
        //auto W = C1*(I1 - 3)  + C1*1000*(I3-2*log(sqrt(I3))-1); // Neo-Hookean
        auto W = C1*(I1 - 3) + C2*pow(I1 - 3, 2) + C3*pow(I1 - 3, 3)  + C1*1000*(I3-2*log(sqrt(I3))-1); // Yeoh
        return W;
    };

    //===========================================================================================
    //
    //===========================================================================================

    auto Area = [](SymMtx3D<> E, unsigned char dif = 3){
        std::array<double, 3> N{0, 0, 1};
        auto I1 = Mech::I1<>(dif, E);
        auto I2 = Mech::I2<>(dif, E);
        double I0NN = 1;
        auto I4NN = Mech::I4fs<>(dif, E, N, N);
        auto I5NN = Mech::I5fs<>(dif, E, N, N);
        
        auto area = sqrt(I0NN*I2 - I4NN*I1 + I5NN);
        return area;
    };
    auto Stos = [Area](const Mtx3D<>& grU)-> double{return Area(Mech::grU_to_E(grU), 2)(); };
    auto dStos = [Area](const Mtx3D<>& grU)-> Mtx3D<>{return Mech::S_to_P(grU, Area(Mech::grU_to_E(grU), 2).D()); }; 
    auto P_func = [Potential](const Mtx3D<>& grU) -> Mtx3D<>{ return Mech::S_to_P(grU, Potential(Mech::grU_to_E(grU), 2).D()); };
    auto dP_func = [Potential](const Mtx3D<>& grU) -> Sym4Tensor3D<> { auto W = Potential(Mech::grU_to_E(grU), 2); return Mech::dS_to_dP(grU, W.D(), W.DD()); };
    auto dJ_func = [](const Mtx3D<>& grU)->Mtx3D<>{ return Mech::S_to_P(grU, Mech::J<>(1, Mech::grU_to_E(grU)).D()); };
    auto ddJ_func = [](const Mtx3D<>& grU)->Sym4Tensor3D<>{ auto J = Mech::J<>(2, Mech::grU_to_E(grU)); return Mech::dS_to_dP(grU, J.D(), J.DD()); };
    auto comp_gradU = [gradUFEM = UFem.getOP(GRAD)](const Coord<> &X, const Tetra<const double>& XYZ, Ani::ArrayView<> udofs, DynMem<>& alloc)->Mtx3D<>{
        Mtx3D<> grU;
        DenseMatrix<> A(grU.m_dat.data(), 9, 1);
        fem3DapplyX(XYZ, ArrayView<const double>(X.data(), 3), DenseMatrix<>(udofs.data, udofs.size, 1), gradUFEM, A, alloc);
        return grU;
    };
    auto comp_U = [UFEMx = UFem.getOP(IDEN)](const Coord<> &X, const Tetra<const double>& XYZ, Ani::ArrayView<> udofs, DynMem<>& alloc)->Coord<>{
        Coord<> Ux;
        DenseMatrix<> A(Ux.data(), 3, 1);
        fem3DapplyX(XYZ, ArrayView<const double>(X.data(), 3), DenseMatrix<>(udofs.data, udofs.size, 1), UFEMx, A, alloc);
        return Ux;
    };
    Mtx3D<> I_2rnk;
    for (unsigned i = 0; i < 3; ++i)
            I_2rnk(i, i) = 1;
    Sym4Tensor3D<> I_4rnk;
    for (unsigned i = 0; i < 3; ++i)
            I_4rnk(i, i, i, i) = 1;
       

    struct BndMarker{
        std::array<int, 4> n = {0};
        std::array<int, 6> e = {0};
        std::array<int, 4> f = {0};

        DofT::TetGeomSparsity getSparsity(int type) const {
            DofT::TetGeomSparsity sp;
            for (int i = 0; i < 4; ++i) if (n[i] & type)
                sp.setNode(i);
            for (int i = 0; i < 6; ++i) if (e[i] & type)
                sp.setEdge(i);  
            for (int i = 0; i < 4; ++i) if (f[i] & type)
                sp.setFace(i);
            return sp;    
        }
        void fillFromBndTag(Tag lbl, Ani::DofT::uint geom_mask, const ElementArray<Node>& nodes, const ElementArray<Edge>& edges, const ElementArray<Face>& faces) {
            if (geom_mask & DofT::NODE){
                for (unsigned i = 0; i < n.size(); ++i) 
                    n[i] = nodes[i].Integer(lbl);
            }
            if (geom_mask & DofT::EDGE){
                for (unsigned i = 0; i < e.size(); ++i) 
                    e[i] = edges[i].Integer(lbl);
            }
            if (geom_mask & DofT::FACE){
                for (unsigned i = 0; i < f.size(); ++i) 
                    f[i] = faces[i].Integer(lbl);
            }
        }
    };

    struct ProbLocData{
        BndMarker lbl;      //< save labels used to apply boundary conditions
        ArrayView<> udofs;  //< save elemental dofs to evaluate grad_j u_i (x)

        //some helper data to be postponed to tensor functions
        const Tetra<const double>* pXYZ = nullptr;    
        DynMem<>* palloc = nullptr;
        const Cell* c = nullptr;
    };

    auto P_tensor = [comp_gradU, P_func](const Coord<> &X, double *D, TensorDims dims, void *user_data, int iTet){
        (void) dims; (void) iTet;
        auto& p = *static_cast<ProbLocData*>(user_data);
        auto grU = comp_gradU(X, *p.pXYZ, p.udofs, *p.palloc);
        auto P = P_func(grU);
        std::copy(P.m_dat.data(), P.m_dat.data() + 9, D);
        return Ani::TENSOR_GENERAL;
    };
    auto dP_tensor = [comp_gradU, dP_func](const Coord<> &X, double *D, TensorDims dims, void *user_data, int iTet){
        (void) dims; (void) iTet;
        auto& p = *static_cast<ProbLocData*>(user_data);
        auto grU = comp_gradU(X, *p.pXYZ, p.udofs, *p.palloc);
        auto dP = tensor_convert<Tensor4Rank<3>>(dP_func(grU));
        std::copy(dP.m_dat.data(), dP.m_dat.data() + 81, D);
        return Ani::TENSOR_SYMMETRIC;
    };
    auto F_tensor = [](const Coord<> &X, double *F, TensorDims dims, void *user_data, int iTet) {
        (void) X; (void) dims; (void) user_data; (void) iTet;
        F[0] = 0, F[1] = 0, F[2] = 0; 
        return Ani::TENSOR_GENERAL;
    };

    auto f_expr = [q, &alpha, r](double dv, int dif = 1){
        auto d = VarPool<1>::Var(0, dv, dif);
        return alpha*(log((1+q)*r/d)/log(1+q))*sq(1+(1-d/r)/q);
    };

    auto tangent_tensor = [q, &alpha, r, &H, comp_U, comp_gradU, Stos, f_expr](const Coord<> &X, double *F, TensorDims dims, void *user_data, int iTet) {
        (void) dims; (void) iTet;
        auto& p = *static_cast<ProbLocData*>(user_data);
        auto U = comp_U(X, *p.pXYZ, p.udofs, *p.palloc);
        auto d = sqrt((X[0]+U[0])*(X[0]+U[0])+(X[1]+U[1])*(X[1]+U[1])+(H-(X[2]+U[2]))*(H-(X[2]+U[2])));
        double sigmoid = 0;
        if(d<(1+q)*r)  sigmoid = f_expr(d, 2)();
        else sigmoid = 0;
        auto grU = comp_gradU(X, *p.pXYZ, p.udofs, *p.palloc);
        double stos = Stos(grU);
        F[0] = -(X[0]+U[0])*sigmoid*stos; 
        F[1] = -(X[1]+U[1])*sigmoid*stos; 
        F[2] = (H-X[2]-U[2])*sigmoid*stos; 
        return Ani::TENSOR_GENERAL;
    };

    auto dtangent_tensor1 = [q, &alpha, r, &H, comp_U, comp_gradU, Stos, f_expr](const Coord<> &X, double *D, TensorDims dims, void *user_data, int iTet) {
        (void) dims; (void) iTet;
        auto& p = *static_cast<ProbLocData*>(user_data);
        auto U = comp_U(X, *p.pXYZ, p.udofs, *p.palloc);
        auto d = sqrt((X[0]+U[0])*(X[0]+U[0])+(X[1]+U[1])*(X[1]+U[1])+(H-(X[2]+U[2]))*(H-(X[2]+U[2])));
        double sd = 0; double sdd = 0;
        if(d<(1+q)*r){sd = f_expr(d, 2)(); sdd = f_expr(d, 2).D()[0];}
        else {sd = 0; sdd = 0;}
        auto grU = comp_gradU(X, *p.pXYZ, p.udofs, *p.palloc);
        double stos = Stos(grU);
        auto du1 = (X[0]+U[0])/d;
        auto du2 = (X[1]+U[1])/d; 
        auto du3 = (X[2]+U[2]-H)/d; 
        D[0] = -sd*stos-(X[0]+U[0])*du1*stos*sdd; 
        D[3] = -(X[0]+U[0])*du2*stos*sdd;
        D[6] = -(X[0]+U[0])*du3*stos*sdd; 
        D[1] = -(X[1]+U[1])*du1*stos*sdd; 
        D[4] = -sd*stos-(X[1]+U[1])*du2*stos*sdd; 
        D[7] = -(X[1]+U[1])*du3*stos*sdd;
        D[2] = (H-X[2]-U[2])*du1*stos*sdd;
        D[5] = (H-X[2]-U[2])*du2*stos*sdd;
        D[8] = -sd*stos+(H-X[2]-U[2])*du3*stos*sdd; 
        return Ani::TENSOR_GENERAL;
    };

    auto dtangent_tensor2 = [q, &alpha, r, &H, comp_U, comp_gradU, dStos, f_expr](const Coord<> &X, double *D, TensorDims dims, void *user_data, int iTet) {
        (void) dims; (void) iTet;
        auto& p = *static_cast<ProbLocData*>(user_data);
        auto U = comp_U(X, *p.pXYZ, p.udofs, *p.palloc);
        auto d = sqrt((X[0]+U[0])*(X[0]+U[0])+(X[1]+U[1])*(X[1]+U[1])+(H-(X[2]+U[2]))*(H-(X[2]+U[2])));
        double sigmoid = 0;
        if(d<(1+q)*r) sigmoid = f_expr(d, 2)();
        else sigmoid = 0;
        auto grU = comp_gradU(X, *p.pXYZ, p.udofs, *p.palloc);
        auto dstos = dStos(grU);
        std::array<double, 3> t;
        t[0] = -(X[0]+U[0])*sigmoid; 
        t[1] = -(X[1]+U[1])*sigmoid; 
        t[2] = (H-X[2]-U[2])*sigmoid;
         
        for (int i = 0; i < 3; ++i)
            for (int p = 0; p < 3; ++p)
                for (int q = 0; q < 3; ++q)
                    D[i + 3*(q + 3*p)   ] = t[i] * dstos(p, q);
        return Ani::TENSOR_GENERAL;
    };

    // Define tag to store result
    Tag u = createFemVarTag(m, *dofmap.target<>(), "u");

    //define function for gathering data from every tetrahedron to send them to elemental assembler
    auto local_data_gatherer = [&BndLabel, geom_mask = UFem.dofMap().GetGeomMask() | DofT::FACE, unf](ElementalAssembler& p) -> void{
        double *nn_p = p.get_nodes();
        const double *args[] = {nn_p, nn_p + 3, nn_p + 6, nn_p + 9};
        ProbLocData data;
        data.lbl.fillFromBndTag(BndLabel, geom_mask, *p.nodes, *p.edges, *p.faces);
        data.udofs.Init(p.vars->begin(0), unf);
        data.c = p.cell;

        p.compute(args, &data);
    };
    std::function<void(const double**, double*, double*, long*, void*, DynMem<double, long>*)> local_jacobian_assembler = 
        [unf, dP_tensor, dtangent_tensor1, dtangent_tensor2, &UFem, order = quad_order](const double** XY/*[4]*/, double* Adat, double* rw, long* iw, void* user_data, DynMem<double, long>* fem_alloc){
        (void) rw, (void) iw;
        DenseMatrix<> A(Adat, unf, unf); A.SetZero();
        auto adapt_alloc = makeAdaptor<double, int>(*fem_alloc);
        auto Bmem = adapt_alloc.alloc(unf*unf, 0, 0);
        DenseMatrix<> B(Bmem.getPlainMemory().ddata, unf, unf); B.SetZero();

        Tetra<const double> XYZ(XY[0], XY[1], XY[2], XY[3]);
        auto& d = *static_cast<ProbLocData*>(user_data);
        d.pXYZ = &XYZ, d.palloc = &adapt_alloc;
        auto grad_u = UFem.getOP(GRAD), iden_u = UFem.getOP(IDEN);

        // elemental stiffness matrix <dP grad(P1), grad(P1)> 
        fem3Dtet<DfuncTraits<>>(XYZ, grad_u, grad_u, dP_tensor, A, adapt_alloc, order, &d);

        // read labels from user_data
        auto& dat = *static_cast<ProbLocData*>(user_data);
        // apply Neumann BC
        for (int k = 0; k < 4; ++k){
            if (dat.lbl.f[k] & PRESSURE_BND){ //Neumann BC
                fem3Dface<DfuncTraits<>> ( XYZ, k, iden_u, iden_u, dtangent_tensor1, B, adapt_alloc, order, &d);
                A += B;
            }
        }
        for (int k = 0; k < 4; ++k){
            if (dat.lbl.f[k] & PRESSURE_BND){ //Neumann BC
                fem3Dface<DfuncTraits<>> ( XYZ, k, iden_u, grad_u, dtangent_tensor2, B, adapt_alloc, order, &d);
                A += B;
            }
        }

        // choose boundary parts of the tetrahedron 
        const std::array<const int, 3> DIR_LABELS = {FIX_X, FIX_Y, FIX_Z};
        for (int d = 0; d < 3; ++d){
            auto sp = dat.lbl.getSparsity(DIR_LABELS[d]);
            //set dirichlet condition
            if (!sp.empty()){
                DofT::NestedDofMapView crdDofMap = UFem.dofMap().GetNestedDofMapView({d});
                applyDirMatrix(crdDofMap, A, sp);
            }
        }  
    };
    std::function<void(const double**, double*, double*, long*, void*, DynMem<double, long>*)> local_residual_assembler = 
        [unf, P_tensor, tangent_tensor, F_tensor, &UFem, grad_u = UFem.getOP(GRAD), iden_u = UFem.getOP(IDEN), order = quad_order](const double** XY/*[4]*/, double* Adat, double* rw, long* iw, void* user_data, DynMem<double, long>* fem_alloc){
        (void) rw, (void) iw;
        DenseMatrix<> F(Adat, unf, 1); F.SetZero();
        auto adapt_alloc = makeAdaptor<double, int>(*fem_alloc);
        auto Bmem = adapt_alloc.alloc(unf, 0, 0);
        DenseMatrix<> B(Bmem.getPlainMemory().ddata, unf, 1); B.SetZero();

        Tetra<const double> XYZ(XY[0], XY[1], XY[2], XY[3]);
        auto& d = *static_cast<ProbLocData*>(user_data);
        d.pXYZ = &XYZ, d.palloc = &adapt_alloc;
        auto grad_u = UFem.getOP(GRAD), iden_u = UFem.getOP(IDEN);  
        ApplyOpFromTemplate<IDEN, FemFix<FEM_P0>> iden_p0;  

        // elemental stiffness matrix <P, grad(P1)> 
        fem3Dtet<DfuncTraits<>>(XYZ, iden_p0, grad_u, P_tensor, F, adapt_alloc, order, &d); 
        // elemental right hand side vector <F, P1>
        fem3Dtet<DfuncTraits<>>(XYZ, iden_p0, iden_u, F_tensor, B, adapt_alloc, order, &d);
        F -= B;

        // read labels from user_data
        auto& dat = *static_cast<ProbLocData*>(user_data);
        // apply Neumann BC
        for (int k = 0; k < 4; ++k){
            if (dat.lbl.f[k] & PRESSURE_BND){ //Neumann BC
                fem3Dface<DfuncTraits<>> ( XYZ, k, iden_p0, iden_u, tangent_tensor, B, adapt_alloc, order, &d);
                F += B;
            }
        }

        // choose boundary parts of the tetrahedron 
        const std::array<const int, 3> DIR_LABELS = {FIX_X, FIX_Y, FIX_Z};
        for (int d = 0; d < 3; ++d){
            auto sp = dat.lbl.getSparsity(DIR_LABELS[d]);
            //set dirichlet condition
            if (!sp.empty()){
                DofT::NestedDofMapView crdDofMap = UFem.dofMap().GetNestedDofMapView({d});
                applyDirResidual(crdDofMap, F, sp);
            }
        } 
    };

    //define assembler
    Assembler discr(m);
    discr.SetMatFunc(GenerateElemMat(local_jacobian_assembler, unf, unf, 0, 0));
    discr.SetRHSFunc(GenerateElemRhs(local_residual_assembler, unf, 0, 0));
    {
        //create global degree of freedom enumenator
        auto Var0Helper = GenerateHelper(*UFem.base());
        FemExprDescr fed;
        fed.PushTrialFunc(Var0Helper, "u");
        fed.PushTestFunc(Var0Helper, "phi_u");
        discr.SetProbDescr(std::move(fed));
    }
    discr.SetDataGatherer(local_data_gatherer);
    discr.PrepareProblem();
    discr.pullInitValFrom(u);

    //get parallel interval and allocate parallel vectors
    auto i0 = discr.getBegInd(), i1 = discr.getEndInd();
    Sparse::Matrix  A( "A" , i0, i1, m->GetCommunicator());
    Sparse::Vector  x( "x" , i0, i1, m->GetCommunicator()), 
                    x0( "x0" , i0, i1, m->GetCommunicator()), 
                    dx("dx", i0, i1, m->GetCommunicator()), 
                    b( "b" , i0, i1, m->GetCommunicator());
    discr.AssembleTemplate(A);  //< preallocate memory for matrix (to accelerate matrix assembling), this call is optional
    // set options to use preallocated matrix state (to accelerate matrix assembling)
    Ani::AssmOpts opts = Ani::AssmOpts().SetIsMtxIncludeTemplate(true)
                                        .SetUseOrderedInsert(true)
                                        .SetIsMtxSorted(true); //< setting this parameters is optional

    //setup linear solver
    Solver lin_solver(p.lin_sol_nm, p.lin_sol_prefix);
    auto assemble_R = [&discr, &u](const Sparse::Vector& x, Sparse::Vector &b) -> int{
        discr.SaveSolution(x, u);
        std::fill(b.Begin(), b.End(), 0.0);
        return discr.AssembleRHS(b);
    };
    auto assemble_J = [&discr, &u, opts](const Sparse::Vector& x, Sparse::Matrix &A) -> int{
        discr.SaveSolution(x, u);
        std::for_each(A.Begin(), A.End(), [](INMOST::Sparse::Row& row){ for (auto vit = row.Begin(); vit != row.End(); ++vit) vit->second = 0.0; });
        return discr.AssembleMatrix(A, opts);
    };
    auto vec_norm = [m](const Sparse::Vector& x)->double{
        double lsum = 0, gsum = 0;
        for (auto itx = x.Begin(); itx != x.End(); ++itx)
            lsum += (*itx) * (*itx);
        gsum = m->Integrate(lsum);    
        // #ifdef USE_MPI
        //     MPI_Allreduce(&lsum, &gsum, 1, MPI_DOUBLE, MPI_SUM, x.GetCommunicator());
        // #else
        //     gsum = lsum;
        // #endif
        return sqrt(gsum);
    };
    auto vec_saxpy = [](double a, const Sparse::Vector& xv, double b, const Sparse::Vector& yv, Sparse::Vector& zv) {
        auto itz = zv.Begin();
        for (auto itx = xv.Begin(), ity = yv.Begin();
            itx != xv.End() && ity != yv.End() && itz != zv.End(); ++itx, ++ity, ++itz)
            *itz = a * (*itx) + b * (*ity);
    };

    auto U_eval = [&discr, &u, unf, &UFem](const Cell& c, const Coord<> &X)->Coord<>{
        auto eval_mreq = fem3DapplyX_memory_requirements(UFem.getOP(IDEN), 1);
        std::vector<char> raw_mem(eval_mreq.enoughRawSize());
        eval_mreq.allocateFromRaw(raw_mem.data(), raw_mem.size());
        std::vector<double> dof_vals(unf);
        DenseMatrix<> dofs(dof_vals.data(), dof_vals.size(), 1);
        discr.GatherDataOnElement(u, c, dofs.data);
        auto nds = c.getNodes();
        reorderNodesOnTetrahedron(nds);
        double XY[4][3] = {0};
        for (int ni = 0; ni < 4; ++ni)
            for (int k = 0; k < 3; ++k)
                XY[ni][k] = nds[ni].Coords()[k];
        Tetra<const double> XYZ(XY[0], XY[1], XY[2], XY[3]);        
        
        Coord<> Ux;
        DenseMatrix<> A(Ux.data(), 3, 1);
        fem3DapplyX( XYZ, ArrayView<const double>(X.data(), 3), dofs, UFem.getOP(IDEN), A, eval_mreq );
        return Ux;
    };

    

    auto grU_eval = [&discr, &u, unf, &UFem](const Cell& c, const Coord<> &X)->Mtx3D<>{
        auto eval_mreq = fem3DapplyX_memory_requirements(UFem.getOP(GRAD), 1);
        std::vector<char> raw_mem(eval_mreq.enoughRawSize());
        eval_mreq.allocateFromRaw(raw_mem.data(), raw_mem.size());
        std::vector<double> dof_vals(unf);
        DenseMatrix<> dofs(dof_vals.data(), dof_vals.size(), 1);
        discr.GatherDataOnElement(u, c, dofs.data);
        auto nds = c.getNodes();
        reorderNodesOnTetrahedron(nds);
        double XY[4][3] = {0};
        for (int ni = 0; ni < 4; ++ni)
            for (int k = 0; k < 3; ++k)
                XY[ni][k] = nds[ni].Coords()[k];
        Tetra<const double> XYZ(XY[0], XY[1], XY[2], XY[3]);        
        
        Mtx3D<> grU;
        DenseMatrix<> A(grU.m_dat.data(), 9, 1);
        fem3DapplyX( XYZ, ArrayView<const double>(X.data(), 3), dofs, UFem.getOP(GRAD), A, eval_mreq);
        return grU;
    };

    auto tangent_comp = [q, &alpha, r, &H, &U_eval, &grU_eval, Stos, f_expr](const Coord<> &X, double *F, TensorDims dims, void *user_data, int iTet) {
        (void) dims; (void) iTet;
        auto& c = *static_cast<Cell*>(user_data);
        auto U = U_eval(c, X);
        auto d = sqrt((X[0]+U[0])*(X[0]+U[0])+(X[1]+U[1])*(X[1]+U[1])+(H-(X[2]+U[2]))*(H-(X[2]+U[2])));
        double sigmoid = 0;
        if(d<(1+q)*r)  sigmoid = f_expr(d, 2)();
        else sigmoid = 0;
        auto grU = grU_eval(c, X);
        double stos = Stos(grU);
        F[0] = (H-X[2]-U[2])*sigmoid*stos; 
        return Ani::TENSOR_GENERAL;
    };

    auto pressure_comp = [&discr, &u, &UFem, unf, &m, &gmsh, &tangent_comp]()->double{
        double p = 0.0;
        BndMarker lbl;   
        ApplyOpFromTemplate<IDEN, FemFix<FEM_P0>> iden_p0;
        auto eval_mreq = fem3Dface_memory_requirements(iden_p0, iden_p0, 2);
        std::vector<char> raw_mem(eval_mreq.enoughRawSize());
        eval_mreq.allocateFromRaw(raw_mem.data(), raw_mem.size());

        for (auto it = m->BeginFace(); it != m->EndFace(); ++it) if ((it->GetStatus() != INMOST::Element::Ghost)&&(it->Boundary())){
            if (!it->HaveData(gmsh) || it->IntegerArray(gmsh).empty()) continue;
            if(it->IntegerArray(gmsh)[0]==4){
                //p += it->Area();
                auto c = it->getCells()[0];

                auto fnds = it->getNodes();
                // auto cnds = c.getNodes();
                // reorderNodesOnTetrahedron(cnds);
                INMOST::ElementArray<Node> cnds(m, 4);
                INMOST::ElementArray<Edge> eds(m, 6); 
                INMOST::ElementArray<Face> fcs(m, 4);
                Ani::collectConnectivityInfo(c, cnds, eds, fcs, true, false);

                int fk = 0;
                for (int i = 0; i < 4; ++i){
                    bool n = true;
                    for (int j = 0; j < 3; ++j){
                        if(fnds[j]==cnds[i]){
                            n = false;
                        } 
                    }
                    if(n) fk = (i+1)%4;
                }


                double X[12]{};
                for (int n = 0; n < 4; ++n)
                   for (int k = 0; k < 3; ++k)
                       X[3*n + k] = cnds[n].Coords()[k];
                double FX[9]{};
                for (int n = 0; n < 3; ++n)
                   for (int k = 0; k < 3; ++k)
                       FX[3*n + k] = fnds[n].Coords()[k];
                Ani::Tetra<const double> XYZ(X+0, X+3, X+6, X+9);
                 
                //std::cout << DenseMatrix<>(X, 3, 4) << " " << fk <<  std::endl;
                //std::cout << DenseMatrix<>(FX, 3, 3) <<  std::endl;

                // double XY[4][3] = {0};
                // for (int ni = 0; ni < 4; ++ni)
                //     for (int k = 0; k < 3; ++k)
                //         XY[ni][k] = cnds[ni].Coords()[k];
                // Tetra<const double> XYZ(XY[0], XY[1], XY[2], XY[3]); 

                ApplyOpFromTemplate<IDEN, FemFix<FEM_P0>> iden_p0;
                double vals[1]{};
                vals[0]=0;
                DenseMatrix<> vm(vals,1, 1);
                fem3Dface<DfuncTraits<>>(XYZ, fk, iden_p0, iden_p0, tangent_comp, vm, eval_mreq, 2, reinterpret_cast<void*>(&c));
                p += abs(vals[0]);
                //std::cout << fk << " " << vals[0] <<  std::endl;
            }
        }
        //p = m->Integrate(p);
        return p;
    };

    auto eval_mreq = fem3DapplyX_memory_requirements(UFem.getOP(IDEN), 1);
    std::vector<char> raw_mem(eval_mreq.enoughRawSize());
    eval_mreq.allocateFromRaw(raw_mem.data(), raw_mem.size());
    std::vector<double> dof_vals(unf);
    DenseMatrix<> dofs(dof_vals.data(), dof_vals.size(), 1);


    auto Inside_eval = [r, &H, &discr, &u, &eval_mreq, &dof_vals, &UFem](const Cell& c, const Coord<> &X)->bool{
        DenseMatrix<> dofs(dof_vals.data(), dof_vals.size(), 1);
        discr.GatherDataOnElement(u, c, dofs.data);
        auto nds = c.getNodes();
        reorderNodesOnTetrahedron(nds);
        double XY[4][3] = {0};
        for (int ni = 0; ni < 4; ++ni)
            for (int k = 0; k < 3; ++k)
                XY[ni][k] = nds[ni].Coords()[k];
        Tetra<const double> XYZ(XY[0], XY[1], XY[2], XY[3]);        
        
        std::array<double,3> Ux;
        DenseMatrix<> A(Ux.data(), 3, 1);
        fem3DapplyX( XYZ, ArrayView<const double>(X.data(), 3), dofs, UFem.getOP(IDEN), A, eval_mreq );
        bool Inside = (((Ux[0]+X[0])*(Ux[0]+X[0]) + (Ux[1]+X[1])*(Ux[1]+X[1]) + (Ux[2]+X[2]-H)*(Ux[2]+X[2]-H)) < r*r);
        return Inside;
    };
    
    auto inside_summ = [&m, Inside_eval](){
    int res = 0;
    for (auto it = m->BeginCell(); it != m->EndCell(); ++it) if (it->GetStatus() != INMOST::Element::Ghost){
        auto nds = it->getNodes();
        double vol = it->Volume();
        std::array<double,3> XY;
        for (int ni = 0; ni < 4; ++ni){
            for (int k = 0; k < 3; ++k)
                XY[k] = nds[ni].Coords()[k];
            bool val = Inside_eval(it->getAsCell(), XY);
            res += val;        
        }        
    }
    //m->Integrate(res);
    return res;
    };
    std::ofstream fout("output.txt");
    int i = 0;
    while(H > Hend+dH)
    {
        H -= dH;
        i += 1;
        if (pRank == 0) std::cout << "Current center height: " << H << std::endl;
        TimerWrap m_timer_total; m_timer_total.reset();
        assemble_R(x, b);
        double anrm = vec_norm(b), rnrm = 1;
        double anrm0 = anrm;
        int ni = 0;
        if (pRank == 0) std::cout << prob_name << ":\n\tnit = " << ni << ": newton residual = " << anrm << " ( rel = " << rnrm << " )" <<  std::endl;
        while (rnrm >= p.nlin_rel_err && anrm >= p.nlin_abs_err && ni < p.nlin_maxit){
            assemble_J(x, A);
            lin_solver.SetMatrix(A);
            if (std::stod(lin_solver.GetParameter("absolute_tolerance")) > p.lin_abs_scale*anrm)
                lin_solver.SetParameterReal("absolute_tolerance", p.lin_abs_scale*anrm);
            lin_solver.Solve(b, dx);
            print_linear_solver_status(lin_solver, prob_name, true);
            vec_saxpy(1, x, -1, dx, x);
            assemble_R(x, b);
            anrm = vec_norm(b);
            rnrm = anrm / anrm0;
            ni++;
            if (pRank == 0) std::cout << prob_name << ":\n\tnit = " << ni << ": newton residual = " << anrm << " ( rel = " << rnrm << " )" <<  std::endl;
        }
        double total_sol_time =  m_timer_total.elapsed();
        if (pRank == 0) std::cout << "Total solution time: " << total_sol_time << "s" << std::endl;
        discr.SaveSolution(x, u);
        if (pRank == 0) std::cout << "Total overlapping nodes: " << inside_summ() << "==== Current alpha:" << alpha <<  std::endl;
        if(inside_summ()>0){
            H += dH;
            i -= 1;
            alpha = alpha*alpha_mul;
            x = x0;
        } else {
            x0 = x;
            m->Save(p.save_dir + p.save_prefix + std::to_string(i/10) + std::to_string(i%10) + ".pvtu");
            if (pRank == 0){fout << H << " " << pressure_comp() <<  std::endl;}
        }
        
    }
    //copy result to the tag and save solution

    discr.Clear();
    InmostFinalize();
    fout.close();

    return 0;
}
