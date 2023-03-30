// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "fuse_expression.h"

#include <algorithm>

namespace pnnx {

static bool operand_maybe_tensor(const Operand* operand)
{
    const Operator* op = operand->producer;

    if (op->type == "prim::Constant")
    {
        const Parameter& param = op->params.at("value");
        if (param.type == 0 || param.type == 1 || param.type == 2 || param.type == 3 || param.type == 4)
        {
            return false;
        }
        else
        {
            return true;
        }
    }

    if (op->type == "prim::NumToTensor")
    {
        return operand_maybe_tensor(op->inputs[0]);
    }

    if (op->type == "prim::ListConstruct")
    {
        return false;
    }

    if (op->type == "aten::size")
    {
        return false;
    }

    if (op->type == "aten::Int")
    {
        return operand_maybe_tensor(op->inputs[0]);
    }

    if (op->type == "aten::to" || op->type == "aten::detach")
    {
        return operand_maybe_tensor(op->inputs[0]);
    }

    if (op->type == "aten::ScalarImplicit")
    {
        return false;
    }

    if (op->type == "aten::abs"
            || op->type == "aten::acos"
            || op->type == "aten::acosh"
            || op->type == "aten::asin"
            || op->type == "aten::asinh"
            || op->type == "aten::atan"
            || op->type == "aten::atanh"
            || op->type == "aten::ceil"
            || op->type == "aten::cos"
            || op->type == "aten::cosh"
            || op->type == "aten::exp"
            || op->type == "aten::floor"
            || op->type == "aten::log"
            || op->type == "aten::neg"
            || op->type == "aten::reciprocal"
            || op->type == "aten::rsqrt"
            || op->type == "aten::sign"
            || op->type == "aten::sin"
            || op->type == "aten::sinh"
            || op->type == "aten::sqrt"
            || op->type == "aten::square"
            || op->type == "aten::tan"
            || op->type == "aten::tanh"
            || op->type == "aten::trunc")
    {
        return operand_maybe_tensor(op->inputs[0]);
    }

    if (op->type == "aten::atan2"
            || op->type == "aten::div"
            || op->type == "aten::floor_divide"
            || op->type == "aten::mul"
            || op->type == "aten::pow")
    {
        return operand_maybe_tensor(op->inputs[0]) || operand_maybe_tensor(op->inputs[1]);
    }

    if (op->type == "aten::__and__" || op->type == "aten::__or__" || op->type == "aten::__xor__")
    {
        return operand_maybe_tensor(op->inputs[0]) || operand_maybe_tensor(op->inputs[1]);
    }

    if (op->type == "aten::add" || op->type == "aten::sub" || op->type == "aten::rsub")
    {
        return operand_maybe_tensor(op->inputs[0]) || operand_maybe_tensor(op->inputs[1]) || operand_maybe_tensor(op->inputs[2]);
    }

    return true;
}

static bool operand_is_foldable(const Operand* operand, const std::set<std::string>& foldable_constants)
{
    if (foldable_constants.find(operand->name) != foldable_constants.end())
        return true;

    const Operator* op = operand->producer;

    if (op->type == "pnnx.Input")
        return false;

    for (auto x : op->inputs)
    {
        if (!operand_is_foldable(x, foldable_constants))
            return false;
    }

    return true;
}

static void fuse_expression(Graph& graph, Operand* operand, std::string& expr, std::vector<Operand*>& inputs, const std::set<std::string>& foldable_constants, bool checksubgraph = true)
{
    // fprintf(stderr, "fuse_expression %s\n", operand->name.c_str());

    Operator* op = operand->producer;

    if (checksubgraph && operand_maybe_tensor(operand))
    {
        if (op->outputs.size() > 1 || op->outputs[0]->consumers.size() > 1)
        {
            auto it = std::find(inputs.begin(), inputs.end(), operand);
            if (it == inputs.end())
            {
                // tensor
                char tmp[32];
                sprintf(tmp, "@%d", (int)inputs.size());
                expr += tmp;

                inputs.push_back(operand);
            }
            else
            {
                // tensor
                char tmp[32];
                sprintf(tmp, "@%d", (int)(it - inputs.begin()));
                expr += tmp;
            }

            return;
        }
    }

    if (op->type == "prim::Constant")
    {
        const Parameter& param = op->params["value"];
        //         fprintf(stderr, "fuse_expression prim::Constant %d\n", param.type);
        if (param.type == 0)
        {
            expr += "None";
        }
        else if (param.type == 1)
        {
            expr += param.b ? "True" : "False";
        }
        else if (param.type == 2)
        {
            char tmp[32];
            sprintf(tmp, "%d", param.i);
            expr += tmp;
        }
        else if (param.type == 3)
        {
            char tmp[32];
            sprintf(tmp, "%e", param.f);
            expr += tmp;
        }
        else if (param.type == 4)
        {
            expr += param.s;
        }
        else
        {
            auto it = std::find(inputs.begin(), inputs.end(), operand);
            if (it == inputs.end())
            {
                // tensor
                char tmp[32];
                sprintf(tmp, "@%d", (int)inputs.size());
                expr += tmp;

                inputs.push_back(operand);
            }
            else
            {
                // tensor
                char tmp[32];
                sprintf(tmp, "@%d", (int)(it - inputs.begin()));
                expr += tmp;
            }
        }
    }
    else if (checksubgraph && operand_maybe_tensor(operand) && operand_is_foldable(operand, foldable_constants))
    {
        // fprintf(stderr, "operand_is_foldable %s\n", operand->name.c_str());

        auto it = std::find(inputs.begin(), inputs.end(), operand);
        if (it == inputs.end())
        {
            // tensor
            char tmp[32];
            sprintf(tmp, "@%d", (int)inputs.size());
            expr += tmp;

            inputs.push_back(operand);
        }
        else
        {
            // tensor
            char tmp[32];
            sprintf(tmp, "@%d", (int)(it - inputs.begin()));
            expr += tmp;
        }
    }
    else if (op->type == "prim::NumToTensor")
    {
        fuse_expression(graph, op->inputs[0], expr, inputs, foldable_constants);
    }
    else if (op->type == "prim::ListConstruct")
    {
        expr += "[";
        for (int i = 0; i < (int)op->inputs.size() - 1; i++)
        {
            fuse_expression(graph, op->inputs[i], expr, inputs, foldable_constants);
            expr += ",";
        }
        if (op->inputs.size() > 0)
        {
            fuse_expression(graph, op->inputs[op->inputs.size() - 1], expr, inputs, foldable_constants);
        }
        expr += "]";
    }
    else if (op->type == "aten::size")
    {
        expr += "size(";
        fuse_expression(graph, op->inputs[0], expr, inputs, foldable_constants);
        expr += ",";
        fuse_expression(graph, op->inputs[1], expr, inputs, foldable_constants);
        expr += ")";
    }
    else if (op->type == "aten::Int")
    {
        expr += "int(";
        fuse_expression(graph, op->inputs[0], expr, inputs, foldable_constants);
        expr += ")";
    }
    else if (op->type == "aten::to" || op->type == "aten::detach" || op->type == "aten::ScalarImplicit")
    {
        fuse_expression(graph, op->inputs[0], expr, inputs, foldable_constants);
    }
    else if (op->type == "aten::abs"
             || op->type == "aten::acos"
             || op->type == "aten::acosh"
             || op->type == "aten::asin"
             || op->type == "aten::asinh"
             || op->type == "aten::atan"
             || op->type == "aten::atanh"
             || op->type == "aten::ceil"
             || op->type == "aten::cos"
             || op->type == "aten::cosh"
             || op->type == "aten::exp"
             || op->type == "aten::floor"
             || op->type == "aten::log"
             || op->type == "aten::neg"
             || op->type == "aten::reciprocal"
             || op->type == "aten::rsqrt"
             || op->type == "aten::sign"
             || op->type == "aten::sin"
             || op->type == "aten::sinh"
             || op->type == "aten::sqrt"
             || op->type == "aten::square"
             || op->type == "aten::tan"
             || op->type == "aten::tanh"
             || op->type == "aten::trunc")
    {
        std::string mathop = op->type.substr(6);

        expr += mathop;
        expr += "(";
        fuse_expression(graph, op->inputs[0], expr, inputs, foldable_constants);
        expr += ")";
    }
    else if (op->type == "aten::atan2"
             || op->type == "aten::div"
             || op->type == "aten::floor_divide"
             || op->type == "aten::mul"
             || op->type == "aten::pow"
             || op->type == "aten::remainder")
    {
        std::string mathop = op->type.substr(6);

        expr += mathop;
        expr += "(";
        fuse_expression(graph, op->inputs[0], expr, inputs, foldable_constants);
        expr += ",";
        fuse_expression(graph, op->inputs[1], expr, inputs, foldable_constants);
        expr += ")";
    }
    else if (op->type == "aten::__and__" || op->type == "aten::__or__" || op->type == "aten::__xor__")
    {
        std::string mathop = op->type.substr(8, 3);
        if (mathop == "or_")
            mathop = "or";

        expr += mathop;
        expr += "(";
        fuse_expression(graph, op->inputs[0], expr, inputs, foldable_constants);
        expr += ",";
        fuse_expression(graph, op->inputs[1], expr, inputs, foldable_constants);
        expr += ")";
    }
    else if (op->type == "aten::add" || op->type == "aten::sub")
    {
        std::string mathop = op->type.substr(6);

        expr += mathop;
        expr += "(";
        fuse_expression(graph, op->inputs[0], expr, inputs, foldable_constants);
        expr += ",";

        std::string expr1;
        std::string expr2;
        fuse_expression(graph, op->inputs[1], expr1, inputs, foldable_constants);
        fuse_expression(graph, op->inputs[2], expr2, inputs, foldable_constants);

        if (expr2 == "1")
        {
            expr += expr1;
        }
        else
        {
            expr += ",";
            expr += "mul(";
            expr += expr1;
            expr += ",";
            expr += expr2;
            expr += ")";
        }

        expr += ")";
    }
    else if (op->type == "aten::rsub")
    {
        expr += "sub(";
        std::string expr1;
        std::string expr2;
        fuse_expression(graph, op->inputs[1], expr1, inputs, foldable_constants);
        fuse_expression(graph, op->inputs[2], expr2, inputs, foldable_constants);

        if (expr2 == "1")
        {
            expr += expr1;
        }
        else
        {
            expr += ",";
            expr += "mul(";
            expr += expr1;
            expr += ",";
            expr += expr2;
            expr += ")";
        }

        expr += ",";
        fuse_expression(graph, op->inputs[0], expr, inputs, foldable_constants);
        expr += ")";
    }
    else
    {
        auto it = std::find(inputs.begin(), inputs.end(), operand);
        if (it == inputs.end())
        {
            // tensor
            char tmp[32];
            sprintf(tmp, "@%d", (int)inputs.size());
            expr += tmp;

            inputs.push_back(operand);
        }
        else
        {
            // tensor
            char tmp[32];
            sprintf(tmp, "@%d", (int)(it - inputs.begin()));
            expr += tmp;
        }
    }
}

void fuse_expression(Graph& graph, const std::set<std::string>& foldable_constants)
{
    int pnnx_expr_index = 0;

    for (;;)
    {
        bool need_fuse = false;

        // build expression via reverse order
        for (int i = (int)graph.ops.size() - 1; i >= 0; i--)
        {
            Operator* op = graph.ops[i];

            if (op->type == "prim::Constant")
            {
                need_fuse = true;
            }
            if (op->type == "prim::NumToTensor")
            {
                need_fuse = true;
            }
            if (op->type == "prim::ListConstruct")
            {
                need_fuse = true;
            }
            if (op->type == "aten::size")
            {
                need_fuse = true;
            }
            if (op->type == "aten::Int")
            {
                need_fuse = true;
            }
            if (op->type == "aten::to" || op->type == "aten::detach" || op->type == "aten::ScalarImplicit")
            {
                need_fuse = true;
            }
            if (op->type == "aten::abs"
                    || op->type == "aten::acos"
                    || op->type == "aten::acosh"
                    || op->type == "aten::add"
                    || op->type == "aten::asin"
                    || op->type == "aten::asinh"
                    || op->type == "aten::atan"
                    || op->type == "aten::atanh"
                    || op->type == "aten::atan2"
                    || op->type == "aten::ceil"
                    || op->type == "aten::cos"
                    || op->type == "aten::cosh"
                    || op->type == "aten::div"
                    || op->type == "aten::exp"
                    || op->type == "aten::floor"
                    || op->type == "aten::floor_divide"
                    || op->type == "aten::log"
                    || op->type == "aten::mul"
                    || op->type == "aten::neg"
                    || op->type == "aten::pow"
                    || op->type == "aten::reciprocal"
                    || op->type == "aten::remainder"
                    || op->type == "aten::rsqrt"
                    || op->type == "aten::rsub"
                    || op->type == "aten::sign"
                    || op->type == "aten::sin"
                    || op->type == "aten::sinh"
                    || op->type == "aten::sqrt"
                    || op->type == "aten::square"
                    || op->type == "aten::sub"
                    || op->type == "aten::tan"
                    || op->type == "aten::tanh"
                    || op->type == "aten::trunc")
            {
                need_fuse = true;
            }
            if (op->type == "aten::__and__" || op->type == "aten::__or__" || op->type == "aten::__xor__")
            {
                need_fuse = true;
            }

            if (need_fuse)
            {
                std::string expr;
                std::vector<Operand*> inputs;
                fuse_expression(graph, op->outputs[0], expr, inputs, foldable_constants, false);
                //                 fprintf(stderr, "expr = %s\n", expr.c_str());

                // lets rewrite graph
                char name[32];
                sprintf(name, "pnnx_expr_%d", pnnx_expr_index++);

                op->type = "pnnx.Expression";
                op->name = name;

                op->params.clear();
                op->attrs.clear();

                op->params["expr"] = expr;

                // fix input output
                for (Operand* operand : op->inputs)
                {
                    operand->consumers.erase(std::find(operand->consumers.begin(), operand->consumers.end(), op));
                }

                op->inputs = inputs;

                for (Operand* operand : op->inputs)
                {
                    operand->consumers.push_back(op);
                }

                break;
            }
        }

        if (!need_fuse)
            break;
    }
}

} // namespace pnnx